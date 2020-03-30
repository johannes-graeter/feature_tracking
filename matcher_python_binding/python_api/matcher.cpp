#include <boost/cstdint.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
//#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#include <viso2/matcher.h>

using namespace viso2;

namespace p = boost::python;
namespace np = p::numpy;

// define function pointer
// void (Matcher::*pushBack1)(uint8_t*, int32_t*, bool) = &Matcher::pushBack;
// void (Matcher::*pushBack2)(uint8_t*, uint8_t*, int32_t*, bool) = &Matcher::pushBack;

///**
// *  Here's the actual converter.  Because we've separated the differences into the above
// functions,
// *  we can write a single template class that works for both matrix2 and vector2.
// */
// template <typename T, int N>
// struct pointer_from_python {

//    /**
//     *  Register the converter.
//     */
//    pointer_from_python() {
//        p::converter::registry::push_back(&convertible, &construct, p::type_id<T>());
//    }

//    /**
//     *  Test to see if we can convert this to the desired type; if not return zero.
//     *  If we can convert, returned pointer can be used by construct().
//     */
//    static void* convertible(PyObject* p) {
//        try {
//            p::object obj(p::handle<>(p::borrowed(p)));
//            std::auto_ptr<np::ndarray> array(new np::ndarray(np::from_object(
//                obj, np::dtype::get_builtin<double>(), N, N, np::ndarray::V_CONTIGUOUS)));
//            if (array->shape(0) != 2)
//                return 0;
//            if (N == 2 && array->shape(1) != 2)
//                return 0;
//            return array.release();
//        } catch (p::error_already_set& err) {
//            p::handle_exception();
//            return 0;
//        }
//    }

//    /**
//     *  Finish the conversion by initializing the C++ object into memory prepared by Boost.Python.
//     */
//    static void construct(PyObject* obj, p::converter::rvalue_from_python_stage1_data* data) {
//        // Extract the array we passed out of the convertible() member function.
//        std::auto_ptr<np::ndarray> array(reinterpret_cast<np::ndarray*>(data->convertible));
//        // Find the memory block Boost.Python has prepared for the result.
//        typedef p::converter::rvalue_from_python_storage<T> storage_t;
//        storage_t* storage = reinterpret_cast<storage_t*>(data);
//        // Use placement new to initialize the result.
//        T* m_or_v = new (storage->storage.bytes) T();
//        // Fill the result with the values from the NumPy array.
//        copy_ndarray_to_mv2(*array, *m_or_v);
//        // Finish up.
//        data->convertible = storage->storage.bytes;
//    }

//};

//// warpper for stereo
// void wrap_matcher_pushBack(np::ndarray const& array0, np::ndarray const& array1, bool replace) {
//    if (array0.get_dtype() != np::dtype::get_builtin<uint8_t>()) {
//        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
//        p::throw_error_already_set();
//    }

//    if (array1.get_dtype() != np::dtype::get_builtin<uint8_t>()) {
//        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
//        p::throw_error_already_set();
//    }

//    if (array0.get_nd() != 2) {
//        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
//        p::throw_error_already_set();
//    }

//    if (array1.get_nd() != 2) {
//        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
//        p::throw_error_already_set();
//    }

//    int32_t row_stride = array0.strides(0) / sizeof(uint8_t);
//    int32_t col_stride = array0.strides(1) / sizeof(uint8_t);

//    if (row_stride != col_stride)
//        throw std::runtime_error("col_stride!=row_stride");
//    if (col_stride != array0.shape(1))
//        throw std::runtime_error("col_stride!=cols");

//    // width, height, stride
//    int32_t dims[] = {array0.shape(1), array0.shape(0), col_stride};
//    pushBack2(reinterpret_cast<uint8_t*>(array0.get_data()),
//                 reinterpret_cast<uint8_t*>(array1.get_data()), dims, replace);
//}

// void    (Matcher::*matchFeatures1)(int32_t, Matrix*) = &Matcher::matchFeatures; //without
// transform prior

// not yet implemented in boost
//    // tell the vector indexing suite not to use operator == since undefined
//    namespace boost { namespace python{namespace indexing {
//    template<>
//    struct value_traits<Matcher::p_match> : public value_traits<int>
//    {
//        static bool const equality_comparable = false;
//        static bool const lessthan_comparable = false;
//    };
//    }}}

///@brief Wrap the push back call of matcher.
//        It requires that the passed
//        NumPy array be exactly what we're looking for - no conversion from nested
//        sequences or arrays with other data types, because we want to modify it
//        in-place. Modified example from boost_1_63_0/libs/python/example/numpy/wrap.cpp
inline void wrapMatcherPushBack(Matcher &obj, np::ndarray const &array, bool replace)
{
    if (array.get_dtype() != np::dtype::get_builtin<uint8_t>())
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }

    int32_t height = array.shape(0);
    int32_t width = array.shape(1);
    //    int32_t row_stride = array.strides(0) / sizeof(uint8_t);
    //    int32_t col_stride = array.strides(1) / sizeof(uint8_t);

    //    if (row_stride != col_stride)
    //        throw std::runtime_error("col_stride!=row_stride");
    //    if (col_stride != width)
    //        throw std::runtime_error("col_stride!=width");

    int32_t dims[] = {width, height, width};

    obj.pushBack(reinterpret_cast<uint8_t *>(array.get_data()), dims, replace);
}

// Converts a C++ vector to a python list
// http://stackoverflow.com/questions/5314319/how-to-export-stdvector
template <class T>
struct VectorToListConverter
{
    static PyObject *convert(const std::vector<T> &vector)
    {
        boost::python::list *l = new boost::python::list();
        for (const auto &el : vector)
        {
            l->append(el);
        }
        return l->ptr();
    }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(matchFeatures_overload, Matcher::matchFeatures, 1, 2)

BOOST_PYTHON_MODULE(matcher)
{
    np::initialize(); // have to put this in any module that uses Boost.NumPy

    p::class_<Matcher::parameters>("MatcherParams")
        .def_readwrite("nms_n", &Matcher::parameters::nms_n)
        .def_readwrite("nms_tau", &Matcher::parameters::nms_tau)
        .def_readwrite("match_binsize", &Matcher::parameters::match_binsize)
        .def_readwrite("match_radius", &Matcher::parameters::match_radius)
        .def_readwrite("match_disp_tolerance", &Matcher::parameters::match_disp_tolerance)
        .def_readwrite("outlier_disp_tolerance", &Matcher::parameters::outlier_disp_tolerance)
        .def_readwrite("outlier_flow_tolerance", &Matcher::parameters::outlier_flow_tolerance)
        .def_readwrite("multi_stage", &Matcher::parameters::multi_stage)
        .def_readwrite("half_resolution", &Matcher::parameters::half_resolution)
        .def_readwrite("refinement", &Matcher::parameters::refinement)
        .def_readwrite("f", &Matcher::parameters::f)
        .def_readwrite("cu", &Matcher::parameters::cu)
        .def_readwrite("cv", &Matcher::parameters::cv)
        .def_readwrite("base", &Matcher::parameters::base);

    p::class_<Matcher::p_match>("Match")
        .def_readwrite("u1p", &Matcher::p_match::u1p)
        .def_readwrite("v1p", &Matcher::p_match::v1p)
        .def_readwrite("i1p", &Matcher::p_match::i1p)
        .def_readwrite("u1c", &Matcher::p_match::u1c)
        .def_readwrite("v1c", &Matcher::p_match::v1c)
        .def_readwrite("i1c", &Matcher::p_match::i1c)
        .def_readwrite("u2p", &Matcher::p_match::u2p)
        .def_readwrite("v2p", &Matcher::p_match::v2p)
        .def_readwrite("i2p", &Matcher::p_match::i2p)
        .def_readwrite("u2c", &Matcher::p_match::u2c)
        .def_readwrite("v2c", &Matcher::p_match::v2c)
        .def_readwrite("i2c", &Matcher::p_match::i2c);

    using Matches = std::vector<Matcher::p_match>;
    //    class_<Matches>("Matches").def(vector_indexing_suite<Matches>());
    //    // "true" because tag_to_noddy has member get_pytye
    //    to_python_converter<Matches, toPythonList<Matcher::p_match>, false>();
    p::to_python_converter<Matches, VectorToListConverter<Matcher::p_match>>();

    p::class_<Matcher, boost::noncopyable>("Matcher", p::init<Matcher::parameters>())
        .def("getMatches", &Matcher::getMatches)
        .def("matchFeatures", &Matcher::matchFeatures, matchFeatures_overload());

    p::def("pushBack", &wrapMatcherPushBack);
}
