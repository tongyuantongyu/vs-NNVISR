//
// Created by TYTY on 2023-01-13 013.
//

#pragma once

#include "NvInfer.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "config.h"

class OptimizationContext {
  OptimizationConfig config;
  nvinfer1::ILogger &logger;
  std::filesystem::path path_prefix;

  nvinfer1::IBuilder *builder;
  nvinfer1::ITimingCache *cache;

  cudaDeviceProp prop;
  size_t total_memory;

  [[nodiscard]] nvinfer1::IBuilderConfig *prepareConfig() const;
  [[nodiscard]] nvinfer1::INetworkDefinition *createNetwork() const;
  int buildFeatureExtract(std::vector<uint8_t> input, const std::filesystem::path& output);
  int buildFeatureFusion(std::vector<uint8_t> input, const std::filesystem::path& output);

 public:
  OptimizationContext(OptimizationConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix);
  int optimize(const std::filesystem::path &folder);
  ~OptimizationContext();
};