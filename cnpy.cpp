// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#include "cnpy.h"
#include <complex>
#include <cstdint>
#include <cstring>
#include <regex>
#include <streambuf>

char cnpy::BigEndianTest()
{
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const std::type_info& t)
{
    if (t == typeid(float))
        return 'f';
    if (t == typeid(double))
        return 'f';
    if (t == typeid(long double))
        return 'f';

    if (t == typeid(int))
        return 'i';
    if (t == typeid(char))
        return 'i';
    if (t == typeid(short))
        return 'i';
    if (t == typeid(long))
        return 'i';
    if (t == typeid(long long))
        return 'i';

    if (t == typeid(unsigned char))
        return 'u';
    if (t == typeid(unsigned short))
        return 'u';
    if (t == typeid(unsigned long))
        return 'u';
    if (t == typeid(unsigned long long))
        return 'u';
    if (t == typeid(unsigned int))
        return 'u';

    if (t == typeid(bool))
        return 'b';

    if (t == typeid(std::complex<float>))
        return 'c';
    if (t == typeid(std::complex<double>))
        return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

template <>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const std::string rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const char* rhs)
{
    // write in little endian
    const size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++)
        lhs.push_back(rhs[byte]);

    return lhs;
}

std::tuple<size_t, std::vector<size_t>, bool> cnpy::parse_npy_header(std::istream& is)
{
    // As of: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format
    using cnpy::constants::numpy_magic;
    using cnpy::constants::numpy_magic_length;
    std::string magic(numpy_magic_length, '\0');
    is.read(&magic[0], numpy_magic_length);
    if (magic != numpy_magic)
        throw std::runtime_error("Not a Numpy file as magic number does not match");

    std::uint8_t major = 0, minor = 0;
    is.read(reinterpret_cast<char*>(&major), sizeof(std::uint8_t));
    is.read(reinterpret_cast<char*>(&minor), sizeof(std::uint8_t));

    std::string header;
    std::uint32_t header_length = 0;  // Note that we may read 16 bits only, need to be sure that everything else is 0
    if ((major == 1) && (minor == 0))
    {
        // Next 2 bytes contain little-endian header length
        is.read(reinterpret_cast<char*>(&header_length), sizeof(std::uint16_t));
        assert(header_length <= std::numeric_limits<std::uint16_t>::max());
        header.resize(header_length);
    }
    else if ((major == 2) && (minor == 0))
    {
        // Next 4 bytes
        is.read(reinterpret_cast<char*>(&header_length), sizeof(std::uint32_t));
        header.resize(header_length);
    }
    else
    {
        throw std::runtime_error("Numpy version " + std::to_string(major) + "." + std::to_string(minor)
                                 + " not supported.");
    }

    if (!is)
        throw std::runtime_error("Cannot read header length");

    is.read(&header[0], static_cast<long>(header.size()));
    if (!is || is.gcount() != static_cast<std::streamsize>(header.size()) || header[header.size() - 1] != '\n')
        throw std::runtime_error("parse_npy_header: failed to read data for header");

    // fortran order
    std::size_t loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: missing 'fortran_order'");

    loc1 += 16;
    const bool fortran_order = (header.substr(loc1, 4) == "True");

    // shape
    loc1 = header.find("(");
    std::size_t loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: missing '(' or ')'");

    const std::regex num_regex("[0-9]+");
    std::smatch sm;
    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);

    std::vector<size_t> shape;
    while (std::regex_search(str_shape, sm, num_regex))
    {
        shape.push_back(std::stoul(sm[0].str()));
        str_shape = sm.suffix();
    }

    // endian, word size, data type
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: missing 'descr'");

    loc1 += 9;
    const bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
    if (!littleEndian)
        throw std::runtime_error("parse_npy_header: endian check failed");

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    const std::size_t word_size = std::stoul(str_ws.substr(0, loc2));

    return std::make_tuple(word_size, std::move(shape), fortran_order);
}

std::tuple<uint16_t, size_t, size_t> cnpy::parse_zip_footer(std::fstream& is)
{
    // Note that this requires the stream to be opened in binary mode, or the behavior is undefined.
    // https://stackoverflow.com/questions/12276834/how-to-read-the-6th-character-from-the-end-of-the-file-ifstream
    using cnpy::constants::zip_footer_length;
    is.seekg(-zip_footer_length, std::ios::end);

    std::vector<char> footer(zip_footer_length);
    is.read(footer.data(), zip_footer_length);
    if (!is || is.gcount() != zip_footer_length)
        throw std::runtime_error("parse_zip_footer: failed to read footer");

    const uint16_t disk_no = *reinterpret_cast<uint16_t*>(&footer[4]);
    const uint16_t disk_start = *reinterpret_cast<uint16_t*>(&footer[6]);
    const uint16_t nrecs_on_disk = *reinterpret_cast<uint16_t*>(&footer[8]);

    const uint16_t nrecs = *reinterpret_cast<uint16_t*>(&footer[10]);
    const size_t global_header_size = *reinterpret_cast<uint32_t*>(&footer[12]);
    const size_t global_header_offset = *reinterpret_cast<uint32_t*>(&footer[16]);
    const uint16_t comment_len = *reinterpret_cast<uint16_t*>(&footer[20]);

    if (disk_no != 0 || disk_start != 0 || nrecs_on_disk != nrecs || comment_len != 0)
        throw std::runtime_error("Unexpected EOCD records");

    return std::make_tuple(nrecs, global_header_size, global_header_offset);
}

cnpy::NpyArray load_the_npy_file(std::ifstream& is)
{
    if (!is || !is.is_open())
        throw std::runtime_error("parse_npy_header: file is not open");

    const auto [word_size, shape, fortran_order] = cnpy::parse_npy_header(is);
    cnpy::NpyArray arr(shape, word_size, fortran_order);
    is.read(arr.data<char>(), static_cast<std::streamsize>(arr.num_bytes()));
    if (!is || is.gcount() != static_cast<std::streamsize>(arr.num_bytes()))
        throw std::runtime_error("load_the_npy_file: failed fread");

    return arr;
}

// https://stackoverflow.com/questions/8815164/c-wrapping-vectorchar-with-istream
template <typename CharT, typename TraitsT = std::char_traits<CharT>>
class vectorwrapbuf : public std::basic_streambuf<CharT, TraitsT>
{
public:
    vectorwrapbuf(std::vector<CharT>& vec)
    {
        this->setg(vec.data(), vec.data(), vec.data() + vec.size());
    }
};

cnpy::NpyArray load_the_npz_array(std::ifstream& is, uint32_t compr_bytes, uint32_t uncompr_bytes)
{
    std::vector<char> buffer_compr(compr_bytes);
    std::vector<char> buffer_uncompr(uncompr_bytes);

    is.read(buffer_compr.data(), compr_bytes);
    if (!is || is.gcount() != compr_bytes)
        throw std::runtime_error("Failed reading compressed data");

    z_stream d_stream;
    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    if (inflateInit2(&d_stream, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("Failed initializing compression system.");

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = reinterpret_cast<unsigned char*>(buffer_compr.data());
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = reinterpret_cast<unsigned char*>(buffer_uncompr.data());

    if (inflate(&d_stream, Z_FINISH) != Z_OK)
        throw std::runtime_error("Failed inflating buffer.");

    if (inflateEnd(&d_stream) != Z_OK)
        throw std::runtime_error("Failed finalizing inflation");

    vectorwrapbuf databuf(buffer_uncompr);
    std::istream header_as_is(&databuf);
    const auto [word_size, shape, fortran_order] = cnpy::parse_npy_header(header_as_is);

    cnpy::NpyArray array(shape, word_size, fortran_order);

    const size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<char>(), buffer_uncompr.data() + offset, array.num_bytes());

    return array;
}

cnpy::npz_t cnpy::npz_load(const std::filesystem::path& fname)
{
    if (!std::filesystem::exists(fname))
        throw std::runtime_error("File does not exist");

    std::ifstream is(fname, std::ios::binary);
    if (!is)
        throw std::runtime_error("Failed opening file");

    cnpy::npz_t arrays;
    using cnpy::constants::zip_header_length;
    while (is)
    {
        std::vector<char> local_header(zip_header_length);
        is.read(&local_header[0], zip_header_length);
        if (!is || is.gcount() != zip_header_length)
            throw std::runtime_error("Failed reading header");

        // If we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04)
            break;

        // Read in the variable name
        const uint16_t name_len = *reinterpret_cast<uint16_t*>(&local_header[26]);
        std::string varname(name_len, ' ');
        is.read(varname.data(), name_len);
        if (!is || is.gcount() != name_len)
            throw std::runtime_error("Failed reading variable name");

        // Erase the trailing .npy
        if (varname.size() >= 4)
            varname.erase(varname.end() - 4, varname.end());

        // Read in the extra field
        const uint16_t extra_field_len = *reinterpret_cast<uint16_t*>(&local_header[28]);
        if (extra_field_len > 0)
        {
            std::vector<char> buff(extra_field_len);
            is.read(buff.data(), extra_field_len);
            if (!is || is.gcount() != extra_field_len)
                throw std::runtime_error("Failed reading extra field");
        }

        const uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[8]);
        const uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[18]);
        const uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[22]);

        if (compr_method == 0)
            arrays[varname] = load_the_npy_file(is);
        else
            arrays[varname] = load_the_npz_array(is, compr_bytes, uncompr_bytes);
    }

    return arrays;
}

cnpy::npz_t npz_load(const std::string& fname) {
  return npz_load(std::filesystem::path(fname));
}

cnpy::NpyArray cnpy::npy_load(const std::filesystem::path& fname)
{
    if (!std::filesystem::exists(fname))
        throw std::runtime_error("File does not exist");

    auto is = std::ifstream(fname, std::ios::binary | std::ios::in);
    if (!is)
        throw std::runtime_error("Cannot open file for reading");

    return load_the_npy_file(is);
}

cnpy::NpyArray npy_load(const std::string& fname) {
  return npy_load(std::filesystem::path(fname));
}
