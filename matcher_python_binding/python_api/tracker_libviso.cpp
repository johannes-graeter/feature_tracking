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
#include <deque>

#include <opencv2/opencv.hpp>
#include <feature_tracking_core/tracklet.h>
#include <feature_tracking_core/tracker_libviso.h>

using namespace feature_tracking;

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
void wrapPushBack(TrackerLibViso &obj, np::ndarray const &array)
{
    if (array.get_dtype() != np::dtype::get_builtin<uint8_t>())
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() > 3 || array.get_nd() < 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }

    int32_t height = array.shape(0);
    int32_t width = array.shape(1);
    int32_t num_channels = array.shape(2);

    cv::Mat img(height, width, CV_8UC(num_channels), reinterpret_cast<uint8_t *>(array.get_data()));
    obj.pushBack(img);
}

void wrapPushBackMask(TrackerLibViso &obj, np::ndarray const& img_array, np::ndarray const& mask_array)
{
    if (img_array.get_dtype() != np::dtype::get_builtin<uint8_t>() || mask_array.get_dtype() != np::dtype::get_builtin<uint8_t>())
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (img_array.get_nd() > 3 || img_array.get_nd() < 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of image dimensions");
        p::throw_error_already_set();
    }
    if (mask_array.get_nd() == 1)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of mask dimensions");
        p::throw_error_already_set();
    }

    int32_t height = img_array.shape(0);
    int32_t width = img_array.shape(1);
    int32_t num_channels = img_array.shape(2);

    if (height != mask_array.shape(0) || width != mask_array.shape(1))
    {
        PyErr_SetString(PyExc_TypeError, "image and mask array have incosistent height or width.");
        p::throw_error_already_set();
    }

    cv::Mat img(height, width, CV_8UC(num_channels), reinterpret_cast<uint8_t *>(img_array.get_data()));
    cv::Mat mask(height, width, CV_8UC1, reinterpret_cast<uint8_t *>(mask_array.get_data()));
    obj.pushBack(img, mask);
}

// Converts a C++ vector to a python list
// http://stackoverflow.com/questions/5314319/how-to-export-stdvector
template <class T>
struct ListConverter                                                                             
{   
    static PyObject *convert(const T& vector)                                               
    {
        boost::python::list *l = new boost::python::list();                                              
        for (const auto &el : vector)                                                                    
        {                                                                                                
            l->append(el);
        }
        return l->ptr();
    }
};  


// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getTrackletsOverload, TrackerLibViso::getTracklets, 1, 2)

BOOST_PYTHON_MODULE(tracker_libviso)
{
    np::initialize(); // have to put this in any module that uses Boost.NumPy

    p::object parameters_class = p::class_<TrackerLibViso::Parameters>("Parameters")
        .def_readwrite("nms_n", &TrackerLibViso::Parameters::nms_n)
        .def_readwrite("nms_tau", &TrackerLibViso::Parameters::nms_tau)
        .def_readwrite("match_binsize", &TrackerLibViso::Parameters::match_binsize)
        .def_readwrite("match_radius", &TrackerLibViso::Parameters::match_radius)
        .def_readwrite("match_disp_tolerance", &TrackerLibViso::Parameters::match_disp_tolerance)
        .def_readwrite("outlier_disp_tolerance", &TrackerLibViso::Parameters::outlier_disp_tolerance)
        .def_readwrite("outlier_flow_tolerance", &TrackerLibViso::Parameters::outlier_flow_tolerance)
        .def_readwrite("multi_stage", &TrackerLibViso::Parameters::multi_stage)
        .def_readwrite("half_resolution", &TrackerLibViso::Parameters::half_resolution)
        .def_readwrite("refinement", &TrackerLibViso::Parameters::refinement)
        .def_readwrite("max_track_length", &TrackerLibViso::Parameters::maxTracklength)
        .def_readwrite("method", &TrackerLibViso::Parameters::method);
    p::object default_parameters = parameters_class();

    p::class_<ImagePoint>("ImagePoint", p::init<>())
	.def(p::init<float, float>())
	.def(p::init<float, float, int>())
	.def_readwrite("index_", &ImagePoint::index_)
	.def_readwrite("u_", &ImagePoint::u_)
	.def_readwrite("v_", &ImagePoint::v_);

    p::class_<Match>("Match", p::init<>())
        .def(p::init<ImagePoint>())
	.def(p::init<float, float>())
	.def(p::init<float, float, int>())
	.def_readwrite("p1_", &Match::p1_);
    
    using MatchDeque = std::deque<Match>;
    p::to_python_converter<MatchDeque, ListConverter<std::deque<Match>>>();

    p::class_<Tracklet, p::bases<MatchDeque>>("Tracklet", p::init<>())
        .def_readwrite("id_", &Tracklet::id_)
        .def_readwrite("age_", &Tracklet::age_);

    using Tracklets = std::vector<Tracklet>;
    //    class_<Matches>("Matches").def(vector_indexing_suite<Matches>());
    //    // "true" because tag_to_noddy has member get_pytye
    //    to_python_converter<Matches, toPythonList<Matcher::p_match>, false>();
    p::to_python_converter<Tracklets, ListConverter<std::vector<Tracklet>>>();

    // Choose one of the overloaded functions and cast to function pointer.
    void (TrackerLibViso::*getTracklets1)(TrackletVector&, int) = &TrackerLibViso::getTracklets;
    p::class_<TrackerLibViso, boost::noncopyable>("TrackerLibViso", p::init<TrackerLibViso::Parameters>())
        .def("get_tracklets", getTracklets1);

    p::def("push_back", &wrapPushBack);
    p::def("push_back_mask", &wrapPushBackMask);
}
