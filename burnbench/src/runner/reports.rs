use comfy_table::{Cell, CellAlignment, Color, Table};
use core::fmt;
use std::{
    fmt::Display,
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use crate::persistence::BenchmarkRecord;

pub(crate) struct FailedBenchmark {
    pub(crate) bench: String,
    pub(crate) backend: String,
}

impl fmt::Display for FailedBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Run the benchmark with verbose enabled to see the error:\ncargo run --bin burnbench -- run --benches {} --backends {} --verbose",
            self.bench, self.backend
        )
    }
}

pub(crate) struct BenchmarkCollection {
    failed_benchmarks: Vec<FailedBenchmark>,
    results_file: PathBuf,
    successful_records: Vec<BenchmarkRecord>,
}

impl Default for BenchmarkCollection {
    fn default() -> Self {
        let results_file = dirs::home_dir()
            .expect("Home directory should exist")
            .join(".cache")
            .join("burn")
            .join("burnbench")
            .join("benchmark_results.txt");
        fs::remove_file(results_file.clone()).ok();
        Self {
            failed_benchmarks: vec![],
            results_file,
            successful_records: vec![],
        }
    }
}

impl BenchmarkCollection {
    pub(crate) fn push_failed_benchmark(&mut self, benchmark: FailedBenchmark) {
        self.failed_benchmarks.push(benchmark);
    }

    pub(crate) fn load_records(&mut self) -> &mut Self {
        if let Ok(file) = fs::File::open(self.results_file.clone()) {
            let file_reader = BufReader::new(file);
            for file in file_reader.lines() {
                let file_path = file.unwrap();
                if let Ok(br_file) = fs::File::open(file_path.clone()) {
                    let benchmarkrecord =
                        serde_json::from_reader::<_, BenchmarkRecord>(br_file).unwrap();
                    self.successful_records.push(benchmarkrecord)
                } else {
                    println!("Cannot find the benchmark-record file: {}", file_path);
                };
            }
        }

        self
    }
}

pub struct ShapeFmt<'a> {
    shapes: &'a Vec<Vec<usize>>,
}

impl<'a> ShapeFmt<'a> {
    pub fn new(shapes: &'a Vec<Vec<usize>>) -> Self {
        Self { shapes }
    }
}

impl Display for ShapeFmt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shapes.is_empty() {
            return f.write_str("()");
        }

        if self.shapes.len() > 1 {
            f.write_str("[")?;
        }

        for shape in self.shapes {
            f.write_str("(")?;
            for (i, val) in shape.iter().enumerate() {
                if i == shape.len() - 1 {
                    f.write_fmt(format_args!("{val}"))?;
                } else {
                    f.write_fmt(format_args!("{val}, "))?;
                }
            }
            f.write_str(")")?;
        }

        if self.shapes.len() > 1 {
            f.write_str("]")?;
        }

        Ok(())
    }
}

impl Display for BenchmarkCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut records = self.successful_records.clone();

        // Sort by benchmark name, then shapes, then median
        records.sort_by(|a, b| {
            a.results
                .name
                .cmp(&b.results.name)
                .then_with(|| a.results.shapes.cmp(&b.results.shapes))
                .then_with(|| {
                    a.results
                        .computed
                        .median
                        .partial_cmp(&b.results.computed.median)
                        .unwrap()
                })
        });

        let mut table = Table::new();
        table.load_preset(comfy_table::presets::ASCII_MARKDOWN);
        table.set_header(vec![
            "Benchmark",
            "Burn Version",
            "Shapes",
            "Feature",
            "Backend",
            "Device",
            "Median",
        ]);

        let mut prev_benchmark = "";
        let mut prev_shapes = vec![];

        // success benchmarks
        for record in &records {
            if prev_benchmark != record.results.name || prev_shapes != record.results.shapes {
                if !prev_benchmark.is_empty() {
                    table.add_row(vec![
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                        Cell::new("----").fg(Color::DarkGrey),
                    ]);
                }
                prev_benchmark = &record.results.name;
                prev_shapes = record.results.shapes.clone();
            }

            table.add_row(vec![
                Cell::new(&record.results.name).fg(Color::Green),
                Cell::new(&record.burn_version).fg(Color::Green),
                Cell::new(format!("{}", ShapeFmt::new(&record.results.shapes))).fg(Color::Green),
                Cell::new(&record.feature).fg(Color::Green),
                Cell::new(format!("`{}`", &record.backend)).fg(Color::Green),
                Cell::new(&record.device).fg(Color::Green),
                Cell::new(format!("{:.3?}", record.results.computed.median))
                    .set_alignment(CellAlignment::Right),
            ]);
        }

        // failed benchmarks
        for benchmark in &self.failed_benchmarks {
            table.add_row(vec![
                Cell::new(&benchmark.bench).fg(Color::Red),
                Cell::new("-"),
                Cell::new("-"),
                Cell::new("-"),
                Cell::new(format!("`{}`", &benchmark.backend)).fg(Color::Red),
                Cell::new("-"),
                Cell::new("FAILED").fg(Color::Red),
            ]);
        }

        write!(f, "{}", table)
    }
}
