#[cfg(feature = "correctness")]
use burn::tensor::Tolerance;
use burn::tensor::{
    Device, Distribution, Int, Tensor, TensorData,
    module::{
        adaptive_avg_pool1d, adaptive_avg_pool2d, avg_pool1d, avg_pool2d, conv_transpose1d,
        conv_transpose2d, conv_transpose3d, conv1d, conv2d, conv2d_weight_backward, conv3d,
        deform_conv2d, interpolate, max_pool1d, max_pool1d_with_indices, max_pool2d,
        max_pool2d_with_indices,
    },
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConvOptions, InterpolateMode, InterpolateOptions,
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

/// Cloneable transport for the prepared input tensors of a relayout op.
#[derive(Clone)]
pub enum BenchmarkInput {
    F3(Vec<Tensor<3>>),
    F4(Vec<Tensor<4>>),
    F5(Vec<Tensor<5>>),
    /// Float tensors plus an integer index tensor (max-pool-with-indices backward).
    F4Idx(Vec<Tensor<4>>, Tensor<4, Int>),
    Conv3 {
        x: Tensor<3>,
        weight: Tensor<3>,
        bias: Tensor<1>,
    },
    Conv4 {
        x: Tensor<4>,
        weight: Tensor<4>,
        bias: Tensor<1>,
    },
    Conv5 {
        x: Tensor<5>,
        weight: Tensor<5>,
        bias: Tensor<1>,
    },
    Deform {
        x: Tensor<4>,
        offset: Tensor<4>,
        weight: Tensor<4>,
    },
    Backward3 {
        x_leaf: Tensor<3>,
        loss: Tensor<1>,
    },
    Backward4 {
        x_leaf: Tensor<4>,
        loss: Tensor<1>,
    },
    BackwardConv3 {
        x_leaf: Tensor<3>,
        weight_leaf: Tensor<3>,
        bias_leaf: Tensor<1>,
        loss: Tensor<1>,
    },
    BackwardConv4 {
        x_leaf: Tensor<4>,
        weight_leaf: Tensor<4>,
        bias_leaf: Tensor<1>,
        loss: Tensor<1>,
    },
    BackwardConv5 {
        x_leaf: Tensor<5>,
        weight_leaf: Tensor<5>,
        bias_leaf: Tensor<1>,
        loss: Tensor<1>,
    },
    BackwardDeform {
        x_leaf: Tensor<4>,
        offset_leaf: Tensor<4>,
        weight_leaf: Tensor<4>,
        loss: Tensor<1>,
    },
}

#[derive(Clone)]
pub enum BenchmarkOutput {
    D1(Tensor<1>),
    D3(Tensor<3>),
    D4(Tensor<4>),
    D5(Tensor<5>),
}

impl BenchmarkOutput {
    pub fn into_data(self) -> TensorData {
        match self {
            BenchmarkOutput::D1(t) => t.into_data(),
            BenchmarkOutput::D3(t) => t.into_data(),
            BenchmarkOutput::D4(t) => t.into_data(),
            BenchmarkOutput::D5(t) => t.into_data(),
        }
    }
}

/// A single relayout operation to benchmark.
trait RelayoutOp {
    fn name(&self) -> String;
    fn shapes(&self) -> Vec<Vec<usize>>;
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput;
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput;
}

/// Optionally prepend an elementwise `+ zeros` op. This is what the relayout
/// optimization fuses the NCHW<->NHWC permutation into.
fn fuse_relayout<const D: usize>(t: Tensor<D>, with_zeros: bool, device: &Device) -> Tensor<D> {
    if with_zeros {
        let zeros = Tensor::zeros(t.shape(), device);
        t + zeros
    } else {
        t
    }
}

/// Random `[N, C, L]` tensor laid out in memory as NLC (the 1D analogue of NHWC).
fn rand_nlc(shape: [usize; 3], device: &Device) -> Tensor<3> {
    let [n, c, l] = shape;
    Tensor::random([n, l, c], Distribution::Default, device).permute([0, 2, 1])
}

/// Random `[N, C, H, W]` tensor laid out in memory as NHWC.
fn rand_nhwc(shape: [usize; 4], device: &Device) -> Tensor<4> {
    let [n, c, h, w] = shape;
    Tensor::random([n, c, h, w], Distribution::Default, device)
}

/// Random `[N, C, D, H, W]` tensor laid out in memory as NDHWC.
fn rand_ndhwc(shape: [usize; 5], device: &Device) -> Tensor<5> {
    let [n, c, d, h, w] = shape;
    Tensor::random([n, d, h, w, c], Distribution::Default, device).permute([0, 4, 1, 2, 3])
}

fn slice_to_string(slice: &[usize]) -> String {
    slice
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("x")
}
struct AvgPool1d {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl RelayoutOp for AvgPool1d {
    fn name(&self) -> String {
        format!(
            "avg_pool1d_k{}_s{}_p{}",
            self.kernel_size, self.stride, self.padding
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D3(avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            false,
            false,
        ))
    }
}

struct AvgPool2d {
    shape: [usize; 4],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl RelayoutOp for AvgPool2d {
    fn name(&self) -> String {
        format!(
            "avg_pool2d_k{}_s{}_p{}",
            slice_to_string(&self.kernel_size),
            slice_to_string(&self.stride),
            slice_to_string(&self.padding)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D4(avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            false,
            false,
        ))
    }
}

struct AdaptiveAvgPool1d {
    shape: [usize; 3],
    output_size: usize,
}

impl RelayoutOp for AdaptiveAvgPool1d {
    fn name(&self) -> String {
        format!("adaptive_avg_pool1d_o{}", self.output_size)
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D3(adaptive_avg_pool1d(x, self.output_size))
    }
}

struct AdaptiveAvgPool2d {
    shape: [usize; 4],
    output_size: [usize; 2],
}

impl RelayoutOp for AdaptiveAvgPool2d {
    fn name(&self) -> String {
        format!(
            "adaptive_avg_pool2d_o{}",
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D4(adaptive_avg_pool2d(x, self.output_size))
    }
}

struct MaxPool1d {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    with_indices: bool,
}

impl RelayoutOp for MaxPool1d {
    fn name(&self) -> String {
        let suffix = if self.with_indices {
            "_with_indices"
        } else {
            ""
        };
        format!(
            "max_pool1d{}_k{}_s{}_p{}_d{}",
            suffix, self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        let out = if self.with_indices {
            max_pool1d_with_indices(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
            .0
        } else {
            max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
        };
        BenchmarkOutput::D3(out)
    }
}

struct MaxPool2d {
    shape: [usize; 4],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    with_indices: bool,
}

impl RelayoutOp for MaxPool2d {
    fn name(&self) -> String {
        let suffix = if self.with_indices {
            "_with_indices"
        } else {
            ""
        };
        format!(
            "max_pool2d{}_k{}_s{}_p{}_d{}",
            suffix,
            slice_to_string(&self.kernel_size),
            slice_to_string(&self.stride),
            slice_to_string(&self.padding),
            slice_to_string(&self.dilation)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        let out = if self.with_indices {
            max_pool2d_with_indices(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
            .0
        } else {
            max_pool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
        };
        BenchmarkOutput::D4(out)
    }
}

struct Interpolate {
    shape: [usize; 4],
    output_size: [usize; 2],
    mode: InterpolateMode,
}

impl RelayoutOp for Interpolate {
    fn name(&self) -> String {
        format!(
            "interpolate_{:?}_o{}",
            self.mode,
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D4(interpolate(
            x,
            self.output_size,
            InterpolateOptions::new(self.mode.clone()),
        ))
    }
}

struct Conv1d {
    x_shape: [usize; 3],
    weight_shape: [usize; 3],
    options: ConvOptions<1>,
}

impl RelayoutOp for Conv1d {
    fn name(&self) -> String {
        format!("conv1d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv3 {
            x: rand_nlc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[0]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv3 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D3(conv1d(x, weight, Some(bias), self.options.clone()))
    }
}

struct Conv2d {
    x_shape: [usize; 4],
    weight_shape: [usize; 4],
    options: ConvOptions<2>,
}

impl RelayoutOp for Conv2d {
    fn name(&self) -> String {
        format!("conv2d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv4 {
            x: rand_nhwc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[0]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv4 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D4(conv2d(x, weight, Some(bias), self.options.clone()))
    }
}

struct Conv3d {
    x_shape: [usize; 5],
    weight_shape: [usize; 5],
    options: ConvOptions<3>,
}

impl RelayoutOp for Conv3d {
    fn name(&self) -> String {
        format!("conv3d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv5 {
            x: rand_ndhwc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[0]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv5 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D5(conv3d(x, weight, Some(bias), self.options.clone()))
    }
}

struct ConvTranspose1d {
    x_shape: [usize; 3],
    weight_shape: [usize; 3],
    options: ConvTransposeOptions<1>,
}

impl RelayoutOp for ConvTranspose1d {
    fn name(&self) -> String {
        format!("conv_transpose1d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv3 {
            x: rand_nlc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[1]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv3 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D3(conv_transpose1d(
            x,
            weight,
            Some(bias),
            self.options.clone(),
        ))
    }
}

struct ConvTranspose2d {
    x_shape: [usize; 4],
    weight_shape: [usize; 4],
    options: ConvTransposeOptions<2>,
}

impl RelayoutOp for ConvTranspose2d {
    fn name(&self) -> String {
        format!("conv_transpose2d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv4 {
            x: rand_nhwc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[1]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv4 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D4(conv_transpose2d(
            x,
            weight,
            Some(bias),
            self.options.clone(),
        ))
    }
}

struct ConvTranspose3d {
    x_shape: [usize; 5],
    weight_shape: [usize; 5],
    options: ConvTransposeOptions<3>,
}

impl RelayoutOp for ConvTranspose3d {
    fn name(&self) -> String {
        format!("conv_transpose3d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Conv5 {
            x: rand_ndhwc(self.x_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
            bias: Tensor::random([self.weight_shape[1]], Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Conv5 { x, weight, bias } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D5(conv_transpose3d(
            x,
            weight,
            Some(bias),
            self.options.clone(),
        ))
    }
}

struct DeformConv2d {
    x_shape: [usize; 4],
    offset_shape: [usize; 4],
    weight_shape: [usize; 4],
    options: DeformConvOptions<2>,
}

impl RelayoutOp for DeformConv2d {
    fn name(&self) -> String {
        format!("deform_conv2d_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.x_shape.to_vec(),
            self.offset_shape.to_vec(),
            self.weight_shape.to_vec(),
        ]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        BenchmarkInput::Deform {
            x: rand_nhwc(self.x_shape, device),
            offset: rand_nhwc(self.offset_shape, device),
            weight: Tensor::random(self.weight_shape, Distribution::Default, device),
        }
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::Deform { x, offset, weight } = input else {
            unreachable!()
        };
        let x = fuse_relayout(x, wz, device);
        let offset = fuse_relayout(offset, wz, device);
        let weight = fuse_relayout(weight, wz, device);
        BenchmarkOutput::D4(deform_conv2d(
            x,
            offset,
            weight,
            None,
            None,
            self.options.clone(),
        ))
    }
}

struct Conv2dWeightBackward {
    x_shape: [usize; 4],
    weight_shape: [usize; 4],
    grad_shape: [usize; 4],
    options: ConvOptions<2>,
}

impl RelayoutOp for Conv2dWeightBackward {
    fn name(&self) -> String {
        format!("conv2d_weight_backward_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, _with_zeros: bool) -> BenchmarkInput {
        let x = rand_nhwc(self.x_shape, device);
        let weight = Tensor::random(self.weight_shape, Distribution::Default, device);
        let output_grad = rand_nhwc(self.grad_shape, device);
        BenchmarkInput::F4(vec![x, weight, output_grad])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, wz: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let output_grad = fuse_relayout(v.pop().unwrap(), wz, device);
        let weight = v.pop().unwrap();
        let x = fuse_relayout(v.pop().unwrap(), wz, device);
        BenchmarkOutput::D4(conv2d_weight_backward(
            x,
            weight,
            output_grad,
            self.options.clone(),
        ))
    }
}

struct AvgPool1dBackward {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl RelayoutOp for AvgPool1dBackward {
    fn name(&self) -> String {
        format!(
            "avg_pool1d_backward_k{}_s{}_p{}",
            self.kernel_size, self.stride, self.padding
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x_leaf = rand_nlc(self.shape, device).require_grad();
        let x = fuse_relayout(x_leaf.clone(), with_zeros, device);
        let loss = avg_pool1d(x, self.kernel_size, self.stride, self.padding, false, false).sum();
        BenchmarkInput::Backward3 { x_leaf, loss }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::Backward3 { x_leaf, loss } = input else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D3(x_leaf.grad(&grads).unwrap())
    }
}

struct AdaptiveAvgPool1dBackward {
    shape: [usize; 3],
    output_size: usize,
}

impl RelayoutOp for AdaptiveAvgPool1dBackward {
    fn name(&self) -> String {
        format!("adaptive_avg_pool1d_backward_o{}", self.output_size)
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x_leaf = rand_nlc(self.shape, device).require_grad();
        let x = fuse_relayout(x_leaf.clone(), with_zeros, device);
        let loss = adaptive_avg_pool1d(x, self.output_size).sum();
        BenchmarkInput::Backward3 { x_leaf, loss }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::Backward3 { x_leaf, loss } = input else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D3(x_leaf.grad(&grads).unwrap())
    }
}

struct AdaptiveAvgPool2dBackward {
    shape: [usize; 4],
    output_size: [usize; 2],
}

impl RelayoutOp for AdaptiveAvgPool2dBackward {
    fn name(&self) -> String {
        format!(
            "adaptive_avg_pool2d_backward_o{}",
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x_leaf = rand_nhwc(self.shape, device).require_grad();
        let x = fuse_relayout(x_leaf.clone(), with_zeros, device);
        let loss = adaptive_avg_pool2d(x, self.output_size).sum();
        BenchmarkInput::Backward4 { x_leaf, loss }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::Backward4 { x_leaf, loss } = input else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D4(x_leaf.grad(&grads).unwrap())
    }
}

struct MaxPool1dWithIndicesBackward {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl RelayoutOp for MaxPool1dWithIndicesBackward {
    fn name(&self) -> String {
        format!(
            "max_pool1d_with_indices_backward_k{}_s{}_p{}_d{}",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x_leaf = rand_nlc(self.shape, device).require_grad();
        let x = fuse_relayout(x_leaf.clone(), with_zeros, device);
        let loss = max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            false,
        )
        .sum();
        BenchmarkInput::Backward3 { x_leaf, loss }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::Backward3 { x_leaf, loss } = input else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D3(x_leaf.grad(&grads).unwrap())
    }
}

struct InterpolateBackward {
    shape: [usize; 4],
    output_size: [usize; 2],
    mode: InterpolateMode,
}

impl RelayoutOp for InterpolateBackward {
    fn name(&self) -> String {
        format!(
            "interpolate_backward_{:?}_o{}",
            self.mode,
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x_leaf = rand_nhwc(self.shape, device).require_grad();
        let x = fuse_relayout(x_leaf.clone(), with_zeros, device);
        let loss = interpolate(
            x,
            self.output_size,
            InterpolateOptions::new(self.mode.clone()),
        )
        .sum();
        BenchmarkInput::Backward4 { x_leaf, loss }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::Backward4 { x_leaf, loss } = input else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D4(x_leaf.grad(&grads).unwrap())
    }
}

#[derive(Clone, Copy)]
enum ConvGrad {
    X,
    Weight,
    Bias,
}

impl ConvGrad {
    fn as_str(&self) -> &'static str {
        match self {
            ConvGrad::X => "x",
            ConvGrad::Weight => "weight",
            ConvGrad::Bias => "bias",
        }
    }
}

struct Conv1dBackward {
    x_shape: [usize; 3],
    weight_shape: [usize; 3],
    options: ConvOptions<1>,
    target: ConvGrad,
}

impl RelayoutOp for Conv1dBackward {
    fn name(&self) -> String {
        format!(
            "conv1d_{}_backward_x{}",
            self.target.as_str(),
            slice_to_string(&self.x_shape)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x = rand_nlc(self.x_shape, device);
        let weight = Tensor::random(self.weight_shape, Distribution::Default, device);
        let bias = Tensor::random([self.weight_shape[0]], Distribution::Default, device);
        let (x, weight, bias) = match self.target {
            ConvGrad::X => (x.require_grad(), weight, bias),
            ConvGrad::Weight => (x, weight.require_grad(), bias),
            ConvGrad::Bias => (x, weight, bias.require_grad()),
        };
        let xf = fuse_relayout(x.clone(), with_zeros, device);
        let wf = fuse_relayout(weight.clone(), with_zeros, device);
        let loss = conv1d(xf, wf, Some(bias.clone()), self.options.clone()).sum();
        BenchmarkInput::BackwardConv3 {
            x_leaf: x,
            weight_leaf: weight,
            bias_leaf: bias,
            loss,
        }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::BackwardConv3 {
            x_leaf,
            weight_leaf,
            bias_leaf,
            loss,
        } = input
        else {
            unreachable!()
        };
        let grads = loss.backward();
        match self.target {
            ConvGrad::X => BenchmarkOutput::D3(x_leaf.grad(&grads).unwrap()),
            ConvGrad::Weight => BenchmarkOutput::D3(weight_leaf.grad(&grads).unwrap()),
            ConvGrad::Bias => BenchmarkOutput::D1(bias_leaf.grad(&grads).unwrap()),
        }
    }
}

struct Conv2dBackward {
    x_shape: [usize; 4],
    weight_shape: [usize; 4],
    options: ConvOptions<2>,
    target: ConvGrad,
}

impl RelayoutOp for Conv2dBackward {
    fn name(&self) -> String {
        format!(
            "conv2d_{}_backward_x{}",
            self.target.as_str(),
            slice_to_string(&self.x_shape)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x = rand_nhwc(self.x_shape, device);
        let weight = Tensor::random(self.weight_shape, Distribution::Default, device);
        let bias = Tensor::random([self.weight_shape[0]], Distribution::Default, device);
        let (x, weight, bias) = match self.target {
            ConvGrad::X => (x.require_grad(), weight, bias),
            ConvGrad::Weight => (x, weight.require_grad(), bias),
            ConvGrad::Bias => (x, weight, bias.require_grad()),
        };
        let xf = fuse_relayout(x.clone(), with_zeros, device);
        let wf = fuse_relayout(weight.clone(), with_zeros, device);
        let loss = conv2d(xf, wf, Some(bias.clone()), self.options.clone()).sum();
        BenchmarkInput::BackwardConv4 {
            x_leaf: x,
            weight_leaf: weight,
            bias_leaf: bias,
            loss,
        }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::BackwardConv4 {
            x_leaf,
            weight_leaf,
            bias_leaf,
            loss,
        } = input
        else {
            unreachable!()
        };
        let grads = loss.backward();
        match self.target {
            ConvGrad::X => BenchmarkOutput::D4(x_leaf.grad(&grads).unwrap()),
            ConvGrad::Weight => BenchmarkOutput::D4(weight_leaf.grad(&grads).unwrap()),
            ConvGrad::Bias => BenchmarkOutput::D1(bias_leaf.grad(&grads).unwrap()),
        }
    }
}

struct Conv3dBackward {
    x_shape: [usize; 5],
    weight_shape: [usize; 5],
    options: ConvOptions<3>,
    target: ConvGrad,
}

impl RelayoutOp for Conv3dBackward {
    fn name(&self) -> String {
        format!(
            "conv3d_{}_backward_x{}",
            self.target.as_str(),
            slice_to_string(&self.x_shape)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.x_shape.to_vec(), self.weight_shape.to_vec()]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x = rand_ndhwc(self.x_shape, device);
        let weight = Tensor::random(self.weight_shape, Distribution::Default, device);
        let bias = Tensor::random([self.weight_shape[0]], Distribution::Default, device);
        let (x, weight, bias) = match self.target {
            ConvGrad::X => (x.require_grad(), weight, bias),
            ConvGrad::Weight => (x, weight.require_grad(), bias),
            ConvGrad::Bias => (x, weight, bias.require_grad()),
        };
        let xf = fuse_relayout(x.clone(), with_zeros, device);
        let wf = fuse_relayout(weight.clone(), with_zeros, device);
        let loss = conv3d(xf, wf, Some(bias.clone()), self.options.clone()).sum();
        BenchmarkInput::BackwardConv5 {
            x_leaf: x,
            weight_leaf: weight,
            bias_leaf: bias,
            loss,
        }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::BackwardConv5 {
            x_leaf,
            weight_leaf,
            bias_leaf,
            loss,
        } = input
        else {
            unreachable!()
        };
        let grads = loss.backward();
        match self.target {
            ConvGrad::X => BenchmarkOutput::D5(x_leaf.grad(&grads).unwrap()),
            ConvGrad::Weight => BenchmarkOutput::D5(weight_leaf.grad(&grads).unwrap()),
            ConvGrad::Bias => BenchmarkOutput::D1(bias_leaf.grad(&grads).unwrap()),
        }
    }
}

struct DeformConv2dBackward {
    x_shape: [usize; 4],
    offset_shape: [usize; 4],
    weight_shape: [usize; 4],
    options: DeformConvOptions<2>,
}

impl RelayoutOp for DeformConv2dBackward {
    fn name(&self) -> String {
        format!("deform_conv2d_backward_x{}", slice_to_string(&self.x_shape))
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.x_shape.to_vec(),
            self.offset_shape.to_vec(),
            self.weight_shape.to_vec(),
        ]
    }
    fn prepare(&self, device: &Device, with_zeros: bool) -> BenchmarkInput {
        let x = rand_nhwc(self.x_shape, device).require_grad();
        let offset = rand_nhwc(self.offset_shape, device).require_grad();
        let weight =
            Tensor::random(self.weight_shape, Distribution::Default, device).require_grad();

        let xf = fuse_relayout(x.clone(), with_zeros, device);
        let of = fuse_relayout(offset.clone(), with_zeros, device);
        let wf = fuse_relayout(weight.clone(), with_zeros, device);
        let loss = deform_conv2d(xf, of, wf, None, None, self.options.clone()).sum();

        BenchmarkInput::BackwardDeform {
            x_leaf: x,
            offset_leaf: offset,
            weight_leaf: weight,
            loss,
        }
    }
    fn run(&self, input: BenchmarkInput, _device: &Device, _with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::BackwardDeform {
            x_leaf,
            offset_leaf: _,
            weight_leaf: _,
            loss,
        } = input
        else {
            unreachable!()
        };
        let grads = loss.backward();
        BenchmarkOutput::D4(x_leaf.grad(&grads).unwrap())
    }
}

pub struct NHWCRelayoutBenchmark {
    device: Device,
    op: Box<dyn RelayoutOp>,
}

impl NHWCRelayoutBenchmark {
    /// Run the op with the relayout fusion (`+ zeros`) and without it, and assert
    /// the outputs match. A mismatch means the fused relayout path is wrong.
    #[cfg(feature = "correctness")]
    fn check_correctness(&self) {
        self.device.seed(42);
        let input_fused = self.op.prepare(&self.device, true);
        let fused = self.op.run(input_fused, &self.device, true);

        self.device.seed(42);
        let input_ref = self.op.prepare(&self.device, false);
        let reference = self.op.run(input_ref, &self.device, false);

        fused
            .into_data()
            .assert_approx_eq(&reference.into_data(), Tolerance::<f32>::balanced());
    }
}

impl Benchmark for NHWCRelayoutBenchmark {
    type Input = BenchmarkInput;
    type Output = BenchmarkOutput;

    fn name(&self) -> String {
        format!(
            "{}_{:?}",
            self.op.name(),
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        self.op.shapes()
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.op.run(input, &self.device, true)
    }

    fn prepare(&self) -> Self::Input {
        self.op.prepare(&self.device, true)
    }

    fn prepare_cloned(&self) -> bool {
        false
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

fn run_ops(ops: Vec<Box<dyn RelayoutOp>>, device: &Device, results: &mut Vec<BenchmarkResult>) {
    for op in ops {
        let bench = NHWCRelayoutBenchmark {
            device: device.clone(),
            op,
        };

        #[cfg(feature = "correctness")]
        bench.check_correctness();

        results.push(run_benchmark(bench));
    }
}

fn bench(
    device: &Device,
    autodiff_device: &Device,
) -> (Vec<BenchmarkResult>, Vec<BenchmarkResult>) {
    let forward_ops: Vec<Box<dyn RelayoutOp>> = vec![
        // Box::new(AvgPool1d {
        //     shape: [2, 4096, 4096],
        //     kernel_size: 4,
        //     stride: 4,
        //     padding: 0,
        // }),
        // Box::new(AvgPool2d {
        //     shape: [2, 64, 512, 512],
        //     kernel_size: [3, 3],
        //     stride: [1, 1],
        //     padding: [1, 1],
        // }),
        // Box::new(AdaptiveAvgPool1d {
        //     shape: [2, 512, 24576],
        //     output_size: 1024,
        // }),
        // Box::new(AdaptiveAvgPool2d {
        //     shape: [2, 64, 512, 512],
        //     output_size: [128, 128],
        // }),
        // Box::new(MaxPool1d {
        //     shape: [2, 5120, 4096],
        //     kernel_size: 4,
        //     stride: 4,
        //     padding: 0,
        //     dilation: 1,
        //     with_indices: false,
        // }),
        // Box::new(MaxPool1d {
        //     shape: [2, 4096, 4096],
        //     kernel_size: 4,
        //     stride: 4,
        //     padding: 0,
        //     dilation: 1,
        //     with_indices: true,
        // }),
        // Box::new(MaxPool2d {
        //     shape: [2, 64, 448, 448],
        //     kernel_size: [3, 3],
        //     stride: [1, 1],
        //     padding: [1, 1],
        //     dilation: [1, 1],
        //     with_indices: false,
        // }),
        // Box::new(MaxPool2d {
        //     shape: [2, 64, 448, 448],
        //     kernel_size: [3, 3],
        //     stride: [1, 1],
        //     padding: [1, 1],
        //     dilation: [1, 1],
        //     with_indices: true,
        // }),
        // Box::new(Interpolate {
        //     shape: [2, 256, 128, 128],
        //     output_size: [384, 384],
        //     mode: InterpolateMode::Nearest,
        // }),
        // Box::new(Interpolate {
        //     shape: [2, 256, 128, 128],
        //     output_size: [384, 384],
        //     mode: InterpolateMode::Bilinear,
        // }),
        // Box::new(Conv1d {
        //     x_shape: [2, 160, 4096],
        //     weight_shape: [160, 160, 3],
        //     options: ConvOptions::new([1], [0], [1], 1),
        // }),
        Box::new(Conv2d {
            x_shape: [2, 64, 128, 128],
            weight_shape: [64, 64, 3, 3],
            options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        }),
        // Box::new(Conv3d {
        //     x_shape: [2, 48, 16, 32, 32],
        //     weight_shape: [48, 48, 3, 3, 3],
        //     options: ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
        // }),
        // Box::new(ConvTranspose1d {
        //     x_shape: [2, 640, 1024],
        //     weight_shape: [640, 640, 3],
        //     options: ConvTransposeOptions::new([1], [0], [0], [1], 1),
        // }),
        // Box::new(ConvTranspose2d {
        //     x_shape: [2, 112, 64, 64],
        //     weight_shape: [112, 112, 3, 3],
        //     options: ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1),
        // }),
        // Box::new(ConvTranspose3d {
        //     x_shape: [2, 20, 16, 16, 16],
        //     weight_shape: [20, 20, 3, 3, 3],
        //     options: ConvTransposeOptions::new([1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
        // }),
        // Box::new(DeformConv2d {
        //     x_shape: [2, 160, 64, 64],
        //     offset_shape: [2, 18, 64, 64],
        //     weight_shape: [160, 160, 3, 3],
        //     options: DeformConvOptions::new([1, 1], [1, 1], [1, 1], 1, 1),
        // }),
        // Box::new(Conv2dWeightBackward {
        //     x_shape: [2, 64, 128, 128],
        //     weight_shape: [64, 64, 3, 3],
        //     grad_shape: [2, 64, 126, 126],
        //     options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        // }),
    ];

    let autodiff_ops: Vec<Box<dyn RelayoutOp>> = vec![
        // Box::new(AvgPool1dBackward {
        //     shape: [2, 1024, 8192],
        //     kernel_size: 4,
        //     stride: 4,
        //     padding: 0,
        // }),
        // Box::new(AdaptiveAvgPool1dBackward {
        //     shape: [2, 640, 8192],
        //     output_size: 1024,
        // }),
        // Box::new(AdaptiveAvgPool2dBackward {
        //     shape: [2, 128, 448, 448],
        //     output_size: [128, 128],
        // }),
        // Box::new(MaxPool1dWithIndicesBackward {
        //     shape: [2, 512, 8192],
        //     kernel_size: 4,
        //     stride: 4,
        //     padding: 0,
        //     dilation: 1,
        // }),
        // Box::new(InterpolateBackward {
        //     shape: [2, 256, 128, 128],
        //     output_size: [320, 320],
        //     mode: InterpolateMode::Nearest,
        // }),
        // Box::new(Conv1dBackward {
        //     x_shape: [2, 352, 4096],
        //     weight_shape: [352, 352, 3],
        //     options: ConvOptions::new([1], [0], [1], 1),
        //     target: ConvGrad::X,
        // }),
        // Box::new(Conv1dBackward {
        //     x_shape: [2, 160, 4096],
        //     weight_shape: [160, 160, 3],
        //     options: ConvOptions::new([1], [0], [1], 1),
        //     target: ConvGrad::Weight,
        // }),
        // Box::new(Conv1dBackward {
        //     x_shape: [2, 512, 8192],
        //     weight_shape: [512, 512, 3],
        //     options: ConvOptions::new([1], [0], [1], 1),
        //     target: ConvGrad::Bias,
        // }),
        // Box::new(Conv2dBackward {
        //     x_shape: [2, 96, 128, 128],
        //     weight_shape: [96, 96, 3, 3],
        //     options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        //     target: ConvGrad::X,
        // }),
        // Box::new(Conv2dBackward {
        //     x_shape: [2, 288, 128, 128],
        //     weight_shape: [288, 288, 3, 3],
        //     options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        //     target: ConvGrad::Bias,
        // }),
        // Box::new(Conv3dBackward {
        //     x_shape: [2, 16, 16, 32, 32],
        //     weight_shape: [16, 16, 3, 3, 3],
        //     options: ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
        //     target: ConvGrad::X,
        // }),
        // Box::new(Conv3dBackward {
        //     x_shape: [2, 28, 16, 32, 32],
        //     weight_shape: [28, 28, 3, 3, 3],
        //     options: ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
        //     target: ConvGrad::Weight,
        // }),
        // Box::new(Conv3dBackward {
        //     x_shape: [2, 64, 16, 32, 32],
        //     weight_shape: [64, 64, 3, 3, 3],
        //     options: ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
        //     target: ConvGrad::Bias,
        // }),
        // Box::new(DeformConv2dBackward {
        //     x_shape: [2, 64, 64, 64],
        //     offset_shape: [2, 18, 64, 64],
        //     weight_shape: [64, 64, 3, 3],
        //     options: DeformConvOptions::new([1, 1], [1, 1], [1, 1], 1, 1),
        // }),
    ];

    let mut forward_results = Vec::new();
    run_ops(forward_ops, device, &mut forward_results);

    let mut backward_results = Vec::new();
    run_ops(autodiff_ops, autodiff_device, &mut backward_results);

    (forward_results, backward_results)
}

fn main() {
    let device = backend_comparison::select_device();
    let autodiff_device = device.clone().autodiff();

    let (forward_results, backward_results) = bench(&device, &autodiff_device);

    backend_comparison::save(forward_results, &device);
    backend_comparison::save(backward_results, &autodiff_device);
}
