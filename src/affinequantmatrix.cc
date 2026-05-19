/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "affinequantmatrix.h"

#include <assert.h>
#include <iostream>
#include <stdexcept>

namespace fasttext {

AffineQuantMatrix::AffineQuantMatrix() : Matrix() {}

AffineQuantMatrix::AffineQuantMatrix(DenseMatrix&& mat)
    : Matrix(mat.size(0), mat.size(1)) {
  data_.resize(m_ * n_);
  row_scales_.resize(m_);
  row_zeros_.resize(m_);
  col_scales_.resize(n_);
  quantize(std::forward<DenseMatrix>(mat));
}

void AffineQuantMatrix::quantize(DenseMatrix&& mat) {
  if (mat.rows() == 0 || mat.cols() == 0) {
    return;
  }

  col_scales_.reserve(cols());
  for (int32_t i = 0; i < cols(); i++) {
    real absmax = std::abs(mat.at(0, i));
    for (int32_t j = 1; j < rows(); j++) {
      absmax = std::max(absmax, std::abs(mat.at(j, i)));
    }
    col_scales_[i] = absmax;
  }

  row_scales_.reserve(rows());
  row_zeros_.reserve(rows());
  data_.reserve(m_ * n_);
  for (int32_t i = 0; i < rows(); i++) {
    real rowmax = mat.at(i, 0) / col_scales_[0];
    real rowmin = mat.at(i, 0) / col_scales_[0];
    for (int32_t j = 1; j < cols(); j++) {
      //real x = mat.at(i, j) / col_max[j];
      real x = mat.at(i, j) / col_scales_[j];
      rowmax = std::max(rowmax, x);
      rowmin = std::min(rowmin, x);
    }
    real s = (rowmax - rowmin) / (qmax - qmin);
    real z = qmin - rowmin / s;
    row_scales_[i] = s;
    row_zeros_[i] = z;
    for (int32_t j = 0; j < cols(); j++) {
      real x = mat.at(i, j) / col_scales_[j];
      uint8_t q = std::round(x / s + z);
      data_[i * cols() + j] = q;
    }
  }
}

Vector AffineQuantMatrix::dequantizeRow(int32_t row) const {
  assert(row >= 0);
  assert(row < m_);
  Vector vec(cols());
  for (int32_t i = 0; i < cols(); i++) {
    // cast from int8 to float
    real q = data_[row * cols() + i];
    real z = row_zeros_[row];
    real s = row_scales_[row];
    real s_col = col_scales_[i];
    vec[i] = (q - z) * s * s_col;
  }
  return vec;
}

real AffineQuantMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  Vector v = dequantizeRow(i);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += v[j] * vec[j];
  }
  if (std::isnan(d)) {
    throw DenseMatrix::EncounteredNaNError();
  }
  return d;
}

void AffineQuantMatrix::addVectorToRow(const Vector&, int64_t, real) {
  throw std::runtime_error("Operation not permitted on quantized matrices.");
}

void AffineQuantMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  Vector v = dequantizeRow(i);
  for (int64_t j = 0; j < n_; j++) {
    x[j] += a * v[j];
  }
}

void AffineQuantMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  Vector v = dequantizeRow(i);
  for (int64_t j = 0; j < n_; j++) {
    x[j] += v[j];
  }
}

void AffineQuantMatrix::averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const {
  x.zero();
  for (auto it = rows.cbegin(); it != rows.cend(); ++it) {
    addRowToVector(x, *it);
  }
  x.mul(1.0 / rows.size());
}

void AffineQuantMatrix::save(std::ostream& out) const {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)&qmin, sizeof(real));
  out.write((char*)&qmax, sizeof(real));
  out.write((char*)col_scales_.data(), col_scales_.size() * sizeof(real));
  out.write((char*)row_scales_.data(), row_scales_.size() * sizeof(real));
  out.write((char*)row_zeros_.data(), row_zeros_.size() * sizeof(real));
  out.write((char*)data_.data(), m_ * n_ * sizeof(uint8_t));
}

void AffineQuantMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  in.read((char*)&qmin, sizeof(real));
  in.read((char*)&qmax, sizeof(real));

  col_scales_.resize(cols());
  row_scales_.resize(rows());
  row_zeros_.resize(rows());
  data_.resize(m_ * n_);

  in.read((char*)col_scales_.data(), cols() * sizeof(real));
  in.read((char*)row_scales_.data(), rows() * sizeof(real));
  in.read((char*)row_zeros_.data(), rows() * sizeof(real));
  in.read((char*)data_.data(), m_ * n_ * sizeof(uint8_t));
}

void AffineQuantMatrix::dump(std::ostream&) const {
  // TODO
  throw std::runtime_error("Not yet implemented.");
}

} // namespace fasttext
