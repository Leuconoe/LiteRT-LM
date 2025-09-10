// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/audio_executor_settings.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/genai_modules/utils/core/ret_check.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

absl::StatusOr<AudioExecutorSettings> AudioExecutorSettings::CreateDefault(
    const ModelAssets& model_assets, int max_sequence_length, Backend backend,
    bool bundled_with_main_model) {
  AudioExecutorSettings settings(model_assets, max_sequence_length);
  RET_CHECK(settings.SetBackend(backend) == absl::OkStatus());
  settings.SetBundledWithMainModel(bundled_with_main_model);
  return settings;
}

int AudioExecutorSettings::GetMaxSequenceLength() const {
  return max_sequence_length_;
}

void AudioExecutorSettings::SetMaxSequenceLength(int max_sequence_length) {
  max_sequence_length_ = max_sequence_length;
}

Backend AudioExecutorSettings::GetBackend() const { return backend_; }

absl::Status AudioExecutorSettings::SetBackend(Backend backend) {
  RET_CHECK(backend == Backend::GPU_ARTISAN || backend == Backend::CPU)
      << "Currently only GPU_ARTISAN and CPU are supported.";
  backend_ = backend;
  return absl::OkStatus();
}

bool AudioExecutorSettings::GetBundledWithMainModel() const {
  return bundled_with_main_model_;
}

void AudioExecutorSettings::SetBundledWithMainModel(
    bool bundled_with_main_model) {
  bundled_with_main_model_ = bundled_with_main_model;
}

}  // namespace litert::lm
