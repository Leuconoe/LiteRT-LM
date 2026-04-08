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

#include "runtime/engine/litert_lm_lib.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert {
namespace lm {
namespace {

using ::nlohmann::json;
using ::testing::status::StatusIs;

TEST(BuildContentListTest, TextOnly) {
  LiteRtLmSettings settings;
  json content_list = json::array();
  std::vector<InputData> input_data;
  input_data.push_back(InputText("Hello world"));
  EXPECT_OK(BuildContentList(input_data, settings, content_list));
  ASSERT_EQ(content_list.size(), 1);
  EXPECT_EQ(content_list[0]["type"], "text");
  EXPECT_EQ(content_list[0]["text"], "Hello world");
}

TEST(BuildContentListTest, MediaTagsSuccess) {
  // Create a temporary file to mock a media file.
  const std::string temp_dir = testing::TempDir();
  const std::string image_path = temp_dir + "/test_image.jpg";
  std::ofstream(image_path) << "dummy image data";

  LiteRtLmSettings settings;
  settings.vision_backend = "cpu";
  json content_list = json::array();

  const std::string prompt =
      absl::StrCat("Describe this [image:", image_path, "].");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  EXPECT_OK(BuildContentList(input_data, settings, content_list));

  ASSERT_EQ(content_list.size(), 3);
  EXPECT_EQ(content_list[0]["type"], "text");
  EXPECT_EQ(content_list[0]["text"], "Describe this ");
  EXPECT_EQ(content_list[1]["type"], "image");
  EXPECT_EQ(content_list[1]["path"], image_path);
  EXPECT_EQ(content_list[2]["type"], "text");
  EXPECT_EQ(content_list[2]["text"], ".");
}

TEST(BuildContentListTest, MediaTagsMissingBackend) {
  const std::string temp_dir = testing::TempDir();
  const std::string image_path = temp_dir + "/test_image.jpg";
  std::ofstream(image_path) << "dummy image data";

  LiteRtLmSettings settings;
  settings.vision_backend = std::nullopt;  // Explicitly missing
  json content_list = json::array();

  std::string prompt = absl::StrCat("Describe this [image:", image_path, "].");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  EXPECT_THAT(BuildContentList(input_data, settings, content_list),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BuildContentListTest, ImageDataSuccess) {
  LiteRtLmSettings settings;
  json content_list = json::array();
  const std::vector<std::string> images = {"image_blob_1", "image_blob_2"};

  std::vector<InputData> input_data;
  input_data.push_back(InputText("Image 1: "));
  input_data.push_back(InputImage(images[0]));
  input_data.push_back(InputText(", Image 2: "));
  input_data.push_back(InputImage(images[1]));
  input_data.push_back(InputText("."));

  EXPECT_OK(BuildContentList(input_data, settings, content_list));

  ASSERT_EQ(content_list.size(), 5);
  EXPECT_EQ(content_list[0]["text"], "Image 1: ");
  EXPECT_EQ(content_list[1]["type"], "image");
  EXPECT_EQ(content_list[1]["blob"], absl::Base64Escape("image_blob_1"));
  EXPECT_EQ(content_list[2]["text"], ", Image 2: ");
  EXPECT_EQ(content_list[3]["type"], "image");
  EXPECT_EQ(content_list[3]["blob"], absl::Base64Escape("image_blob_2"));
  EXPECT_EQ(content_list[4]["text"], ".");
}

TEST(BuildContentListTest, MixedModality) {
  const std::string temp_dir = testing::TempDir();
  const std::string audio_path = temp_dir + "/test_audio.wav";
  std::ofstream(audio_path) << "dummy audio data";

  LiteRtLmSettings settings;
  settings.audio_backend = "cpu";
  json content_list = json::array();
  std::vector<std::string> images = {"image_blob_1"};

  std::string prompt =
      absl::StrCat("Listen to [audio:", audio_path, "] and look at ");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  input_data.push_back(InputImage(images[0]));
  input_data.push_back(InputText("."));
  EXPECT_OK(BuildContentList(input_data, settings, content_list));

  ASSERT_EQ(content_list.size(), 5);
  EXPECT_EQ(content_list[0]["text"], "Listen to ");
  EXPECT_EQ(content_list[1]["type"], "audio");
  EXPECT_EQ(content_list[1]["path"], audio_path);
  EXPECT_EQ(content_list[2]["text"], " and look at ");
  EXPECT_EQ(content_list[3]["type"], "image");
  EXPECT_EQ(content_list[3]["blob"], absl::Base64Escape("image_blob_1"));
  EXPECT_EQ(content_list[4]["text"], ".");
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithEmptyModelPathReturnsError) {
  LiteRtLmSettings settings;
  settings.model_path = "";
  EXPECT_THAT(RunLiteRtLm(settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Following tests are for various model file metadata and tokenizer types.
// They are not exhaustive, but designed to test a variety of scenarios.
// If metadata or tokenizer types are not handled properly, these tests could
// fail.
TEST(LiteRtLmLibTest, RunLiteRtLmWithValidModelPath) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithInferredGemma3ModelType) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_no_model_type.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithDeepseekMetadataTokenizer) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_deepseek_metadata_tokenizer.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}

}  // namespace
}  // namespace lm
}  // namespace litert
