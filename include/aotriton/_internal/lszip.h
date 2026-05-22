// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_LSZIP_H
#define AOTRITON_V3_LSZIP_H

#include <aotriton/config.h>
#include <cstdint>
#include <functional>
#include <string_view>

namespace AOTRITON_NS {

// Parse the central directory of a STORED (uncompressed) ZIP file and invoke
// visitor(entry_name, data_offset, data_size) for each entry.
// fd must be an open, seekable file descriptor; the caller retains ownership.
void lszip(int fd,
           std::function<void(std::string_view name,
                              uint64_t data_offset,
                              uint64_t data_size)> visitor);

} // namespace AOTRITON_NS

#endif // AOTRITON_V3_LSZIP_H
