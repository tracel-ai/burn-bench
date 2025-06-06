use burn::{
    backend::Autodiff,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::{Element, activation::softmax, backend::AutodiffBackend},
};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};

#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone)]
pub struct InferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Config)]
pub struct ModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        Model {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
        .to_device(device)
    }
}

/// Define model behavior
impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        item: TrainingBatch<B>,
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();

        let device = item.tokens.device();
        let tokens = item.tokens;
        let labels = item.labels;
        let mask_pad = item.mask_pad;

        // Calculate token and position embeddings, and combine them
        let index_positions = Tensor::arange(0..seq_length as i64, &device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Perform transformer encoding, calculate output and loss
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        (loss, output_classification, labels)
    }

    /// Defines forward pass for inference
    pub fn infer(&self, item: InferenceBatch<B>) -> Tensor<B, 2> {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();

        let device = item.tokens.device();
        let tokens = item.tokens;
        let mask_pad = item.mask_pad;

        // Calculate token and position embeddings, and combine them
        let index_positions = Tensor::arange(0..seq_length as i64, &device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Perform transformer encoding, calculate output and apply softmax for prediction
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }
}

pub struct TransformerEncoderBenchmark<B: Backend, const AD: bool> {
    shape: Shape,
    device: B::Device,
    config: ModelConfig,
}

impl<B: AutodiffBackend> Benchmark for TransformerEncoderBenchmark<B, true> {
    type Input = (Model<B>, TrainingBatch<B>);
    type Output = B::Gradients;

    fn name(&self) -> String {
        format!("transformer-encoder-training-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (model, input): Self::Input) -> Self::Output {
        let (loss, ..) = model.forward(input);
        loss.backward()
    }

    fn prepare(&self) -> Self::Input {
        (
            self.config.init(&self.device),
            TrainingBatch {
                tokens: Tensor::arange(0..self.shape.num_elements() as i64, &self.device)
                    .reshape(self.shape.clone()),
                labels: Tensor::arange(0..self.shape.dims[0] as i64, &self.device),
                mask_pad: Tensor::<B, 2>::zeros(self.shape.clone(), &self.device).equal_elem(0.0),
            },
        )
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

impl<B: Backend> Benchmark for TransformerEncoderBenchmark<B, false> {
    type Input = (Model<B>, InferenceBatch<B>);
    type Output = Tensor<B, 2>;

    fn name(&self) -> String {
        format!("transformer-encoder-inference-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (model, input): Self::Input) -> Self::Output {
        model.infer(input)
    }

    fn prepare(&self) -> Self::Input {
        (
            self.config.init(&self.device),
            InferenceBatch {
                tokens: Tensor::arange(0..self.shape.num_elements() as i64, &self.device)
                    .reshape(self.shape.clone()),
                mask_pad: Tensor::<B, 2>::zeros(self.shape.clone(), &self.device).equal_elem(0.0),
            },
        )
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    // Something similar to RoBERTa-base.
    let config = ModelConfig::new(
        TransformerEncoderConfig::new(768, 3072, 12, 12).with_norm_first(true),
        10,
        50_265,
        512,
    );

    let batch_size = 2;
    let sequence_length = 256;
    let shape = [batch_size, sequence_length];
    let benchmark_inference = TransformerEncoderBenchmark::<B, false> {
        shape: shape.into(),
        device: device.clone(),
        config: config.clone(),
    };
    let benchmark_training = TransformerEncoderBenchmark::<Autodiff<B>, true> {
        shape: shape.into(),
        device: device.clone(),
        config,
    };

    vec![
        run_benchmark(benchmark_inference),
        run_benchmark(benchmark_training),
    ]
}

fn main() {
    burnbench::bench_on_backend!();
}
