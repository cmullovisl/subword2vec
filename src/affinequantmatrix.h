/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

#include <memory>
#include <vector>
#include <cmath>

#include "real.h"

#include "densematrix.h"
#include "matrix.h"
#include "vector.h"

namespace fasttext {

class AffineQuantMatrix : public Matrix {
 protected:
  std::vector<uint8_t> data_;
  std::vector<real> col_scales_;
  std::vector<real> row_scales_;
  std::vector<real> row_zeros_;
  real qmin = 0.0;
  real qmax = 255.0;

 public:
  AffineQuantMatrix();
  AffineQuantMatrix(DenseMatrix&&);
  AffineQuantMatrix(const AffineQuantMatrix&) = delete;
  AffineQuantMatrix(AffineQuantMatrix&&) = delete;
  AffineQuantMatrix& operator=(const AffineQuantMatrix&) = delete;
  AffineQuantMatrix& operator=(AffineQuantMatrix&&) = delete;
  virtual ~AffineQuantMatrix() noexcept override = default;

  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }

  void quantize(DenseMatrix&& mat);
  Vector dequantizeRow(int32_t row) const;

  real dotRow(const Vector&, int64_t) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addVectorToRow(const Vector&, int64_t, const DenseMatrix&, int32_t) {};
  void addRowToVector(Vector& x, int32_t i) const override;
  void addRowToVector(Vector& x, int32_t i, real a) const override;
  void addRowToVector(Vector& x, int32_t i, const DenseMatrix& W, int32_t k) const {};
  void averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const override;
  void averageRowsTimesWeightsToVector(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix& weights, const std::vector<int32_t>& pos) const {};
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
};

} // namespace fasttext
