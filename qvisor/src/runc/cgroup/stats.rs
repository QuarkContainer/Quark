// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");

use std::fs;

use containerd_shim::protos::cgroups::metrics::{
    CPUStat, CPUUsage, MemoryEntry, MemoryStat, Metrics,
};

use super::super::super::qlib::common::*;

use super::cgroup::Cgroup;

fn read_u64_file(path: &str) -> Result<u64> {
    let s = fs::read_to_string(path)
        .map_err(|e| Error::IOError(format!("read {}: {:?}", path, e)))?;
    let s = s.trim();
    if s == "max" {
        return Ok(0);
    }
    s.parse::<u64>()
        .map_err(|e| Error::IOError(format!("parse {}: {:?}", path, e)))
}

fn read_cpu_usage_usec(cpu_path: &str) -> Result<u64> {
    let stat_path = format!("{}/cpu.stat", cpu_path);
    let contents = fs::read_to_string(&stat_path)
        .map_err(|e| Error::IOError(format!("read {}: {:?}", stat_path, e)))?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("usage_usec ") {
            return rest
                .trim()
                .parse::<u64>()
                .map_err(|e| Error::IOError(format!("parse cpu.stat: {:?}", e)));
        }
    }
    Ok(0)
}

pub fn MetricsFromCgroup(cgroup: &Cgroup) -> Result<Metrics> {
    let mut metrics = Metrics::new();

    let mem_path = cgroup.MakePath("memory");
    let usage = read_u64_file(&format!("{}/memory.current", mem_path))
        .or_else(|_| read_u64_file(&format!("{}/memory.usage_in_bytes", mem_path)))?;

    let mut mem_entry = MemoryEntry::new();
    mem_entry.set_usage(usage);
    let mut mem_stat = MemoryStat::new();
    mem_stat.set_usage(mem_entry);
    metrics.set_memory(mem_stat);

    let cpu_path = cgroup.MakePath("cpu");
    let usage_usec = read_cpu_usage_usec(&cpu_path).unwrap_or(0);
    let mut cpu_usage = CPUUsage::new();
    cpu_usage.set_total(usage_usec);
    let mut cpu_stat = CPUStat::new();
    cpu_stat.set_usage(cpu_usage);
    metrics.set_cpu(cpu_stat);

    Ok(metrics)
}
