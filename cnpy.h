// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include "numpy_defs.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <zlib.h>

namespace cnpy
{

struct NpyArray
{
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order)
          : shape(_shape)
          , word_size(_word_size)
          , fortran_order(_fortran_order)
          , num_vals(1)
    {
        for (size_t i = 0; i < shape.size(); i++)
            num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(new std::vector<char>(num_vals * word_size));
    }

    NpyArray()
          : shape(0)
          , word_size(0)
          , fortran_order(0)
          , num_vals(0)
    {
    }

    template <typename T>
    T* data()
    {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template <typename T>
    const T* data() const
    {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template <typename T>
    std::vector<T> as_vec() const
    {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const
    {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

char BigEndianTest();
char map_type(const std::type_info& t);
std::tuple<size_t, std::vector<size_t>, bool> parse_npy_header(std::istream& file);
std::tuple<uint16_t, size_t, size_t> parse_zip_footer(std::fstream& file);
npz_t npz_load(const std::filesystem::path& fname);
NpyArray npy_load(const std::filesystem::path& fname);
// String-based overloads (compatibility)
npz_t    npz_load(const std::string& fname);
NpyArray npy_load(const std::string& fname);
  
template <typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs)
{
    // write in little endian
    for (size_t byte = 0; byte < sizeof(T); byte++)
    {
        char val = *((char*)&rhs + byte);
        lhs.push_back(val);
    }
    return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);

template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape)
{
    using cnpy::constants::numpy_magic;
    using cnpy::constants::numpy_magic_length;
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); i++)
    {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1)
        dict += ",";

    dict += "), }";
    // Add padding to align at 64. We have: magic string, 2 bytes for the version,
    // 2 or 4 bytes for the length, the dictionary itself the padding and the trailing `\n`
    // Check length of dict (with extra \n, if too big will switch to version 2.0
    const auto big_header = dict.size() + 1 > std::numeric_limits<uint16_t>::max();
    const std::size_t remainder = 64 - (numpy_magic_length + (big_header ? 4 : 2) + dict.size() + 1) % 64;
    dict.insert(dict.end(), remainder, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += numpy_magic;
    header += (big_header ? (char)0x02 : (char)0x01);  // major version of numpy format
    header += (char)0x00;                              // minor version of numpy format
    if (big_header)
        header += (uint32_t)dict.size();
    else
        header += (uint16_t)dict.size();

    header.insert(header.end(), dict.begin(), dict.end());

    return header;
}

template <typename T>
void npy_save(const std::filesystem::path& fname, const T* data, const std::vector<size_t>& shape,const std::string& mode = "w")
{
    std::fstream file;
    std::vector<size_t> true_data_shape;

    if (std::filesystem::exists(fname) && mode == "a")
    {
        file.open(fname, std::ios::in | std::ios::out | std::ios::binary);
        if (!file)
            throw std::runtime_error("Error opening file");

        // File exists, append mode
        const auto [word_size, ds, fortran_order] = parse_npy_header(file);
        true_data_shape = ds;
        if (!fortran_order)
            throw std::runtime_error("Appending to existing fortran-order file not supported");

        if (word_size != sizeof(T))
            throw std::runtime_error("Appending to existing file with wrong word size: new=" + std::to_string(sizeof(T))
                                     + " existing=" + std::to_string(word_size));

        if (true_data_shape.size() != shape.size())
            throw std::runtime_error("Appending to existing file with wrong shape: new=" + std::to_string(shape.size())
                                     + " existing=" + std::to_string(true_data_shape.size()));

        for (size_t i = 1; i < shape.size(); i++)
        {
            if (shape[i] != true_data_shape[i])
            {
                throw std::runtime_error("Appending to existing data with wrong dimension: new="
                                         + std::to_string(shape[i])
                                         + " existing=" + std::to_string(true_data_shape[i]));
            }
        }
        true_data_shape[0] += shape[0];
    }
    else
    {
        // Create a new file
        file.open(fname, std::ios::out | std::ios::binary);
        true_data_shape = shape;
    }

    if (!file.is_open())
        throw std::runtime_error("Cannot open file for writing");

    const std::vector<char> header = create_npy_header<T>(true_data_shape);
    const size_t nels = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    file.seekp(0, std::ios::beg);
    // In append mode we have checked the shape of data are the same, the header has the same dimension
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.seekp(0, std::ios::end);
    file.write(reinterpret_cast<const char*>(data), sizeof(T) * nels);
    file.close();
}

template <typename T>  
void npy_save(const std::string& fname, const T* data, const std::vector<size_t>& shape,const std::string& mode = "w")
{
  return npy_save(std::filesystem::path(fname), data, shape, mode);
}
  
template <typename T>
void npz_save(const std::filesystem::path& zipname, std::string key, const T* data, const std::vector<size_t>& shape, const std::string& mode = "w")              
{
    // first, append a .npy to the key (it is a file in a zip)
    key += ".npy";

    std::fstream ios;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    std::vector<char> global_header;

    if (std::filesystem::exists(zipname) && mode == "a")
    {
        ios.open(zipname, std::ios::binary | std::ios::in | std::ios::out);
        if (!ios)
            throw std::runtime_error("Error opening file");

        // zip file exists. we need to add a new npy file to it.
        size_t global_header_size = 0;
        ///////parse_zip_footer(ios, nrecs, global_header_size, global_header_offset);
        std::tie(nrecs, global_header_size, global_header_offset) = parse_zip_footer(ios);
        ios.seekg(static_cast<long>(global_header_offset), std::ios::beg);
        global_header.resize(global_header_size);
        ios.read(global_header.data(), static_cast<long>(global_header_size));
        if (ios.gcount() != static_cast<std::streamsize>(global_header_size))
            throw std::runtime_error("npz_save: header read error while adding to existing zip");

        // Rewind at beginning of global header offset
        ios.seekp(static_cast<long>(global_header_offset), std::ios::beg);
        if (!ios)
            throw std::runtime_error("Cannot seek at correct position");
    }
    else
    {
        ios.open(zipname, std::ios::binary | std::ios::out);
    }

    if (!ios.is_open())
        throw std::runtime_error("npz_save: failed to open file for writing");

    const std::vector<char> npy_header = create_npy_header<T>(shape);
    const size_t nels = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    const size_t nbytes = nels * sizeof(T) + npy_header.size();

    uint32_t crc =
        crc32(0L, reinterpret_cast<const uint8_t*>(&npy_header[0]), static_cast<unsigned int>(npy_header.size()));
    crc = crc32(crc, reinterpret_cast<const uint8_t*>(data), static_cast<unsigned int>(nels * sizeof(T)));

    std::vector<char> local_header;
    local_header += "PK";                  // first part of sig
    local_header += (uint16_t)0x0403;      // second part of sig
    local_header += (uint16_t)20;          // min version to extract
    local_header += (uint16_t)0;           // general purpose bit flag
    local_header += (uint16_t)0;           // compression method
    local_header += (uint16_t)0;           // file last mod time
    local_header += (uint16_t)0;           // file last mod date
    local_header += (uint32_t)crc;         // crc
    local_header += (uint32_t)nbytes;      // compressed size
    local_header += (uint32_t)nbytes;      // uncompressed size
    local_header += (uint16_t)key.size();  // fname length
    local_header += (uint16_t)0;           // extra field length
    local_header += key;

    // build global header
    global_header += "PK";              // first part of sig
    global_header += (uint16_t)0x0201;  // second part of sig
    global_header += (uint16_t)20;      // version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    global_header += (uint16_t)0;                     // file comment length
    global_header += (uint16_t)0;                     // disk number where file starts
    global_header += (uint16_t)0;                     // internal file attributes
    global_header += (uint32_t)0;                     // external file attributes
    global_header += (uint32_t)global_header_offset;  // relative offset of local file header, since it begins where the
                                                      // global header used to begin
    global_header += key;

    // build footer
    std::vector<char> footer;
    footer += "PK";                            // first part of sig
    footer += (uint16_t)0x0605;                // second part of sig
    footer += (uint16_t)0;                     // number of this disk
    footer += (uint16_t)0;                     // disk where footer starts
    footer += (uint16_t)(nrecs + 1);           // number of records on this disk
    footer += (uint16_t)(nrecs + 1);           // total number of records
    footer += (uint32_t)global_header.size();  // nbytes of global headers
    footer += (uint32_t)(global_header_offset + nbytes
                         + local_header.size());  // offset of start of global headers, since global header now starts
                                                  // after newly written array
    footer += (uint16_t)0;                        // zip file comment length

    ios.write(reinterpret_cast<const char*>(local_header.data()), static_cast<std::streamsize>(local_header.size()));
    ios.write(reinterpret_cast<const char*>(npy_header.data()), static_cast<std::streamsize>(npy_header.size()));
    ios.write(reinterpret_cast<const char*>(data), nels * sizeof(T));
    ios.write(reinterpret_cast<const char*>(global_header.data()), static_cast<std::streamsize>(global_header.size()));
    ios.write(reinterpret_cast<const char*>(footer.data()), static_cast<std::streamsize>(footer.size()));
    ios.close();
}

template <typename T>
void npz_save(const std::string& zipname, std::string key, const T* data, const std::vector<size_t>& shape, const std::string& mode = "w")
{
  return npz_save(std::filesystem::path(zipname), key, data, shape, mode);
}

  
}  // namespace cnpy

#endif
