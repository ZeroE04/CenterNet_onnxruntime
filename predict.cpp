#include <core/session/onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <core/session/onnxruntime_c_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <math.h>
 
using namespace cv;
using namespace std;
 

 
#define PI 3.14159265358979323846
 
struct Box
{
    float x1;
    float y1;
    float x2;
    float y2;
    float ang;
};
 
struct Detection
{
    Box bbox;
    int classId;
    float prob;
};

float Logist(float data)
{
    return 1. / (1. + exp(-data));
}
 
void preProcess(cv::Mat &img, float *output)
{
    int input_w = 512;
    int input_h = 512;
    float scale = cv::min(float(input_w) / img.cols, float(input_h) / img.rows);
    auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);
 
    cv::Mat resized;
    cv::resize(img, resized, scaleSize, 0, 0);
    cv::Mat cropped = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    cv::Rect rect((input_w - scaleSize.width) / 2, (input_h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
 
    resized.copyTo(cropped(rect));
    // imwrite("img_process.png", cropped); 
 
    constexpr static float mean[] = {0.5194416012442385, 0.5378052387430711, 0.533462090585746};
    constexpr static float std[] = {0.3001546018824507, 0.28620901391179554, 0.3014112676161966};
    int row = 512;
    int col = 512;
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = cropped.ptr<uchar>(i)[j * 3 + c];
                output[c * row * col + i * col + j] = (pix / 255. - mean[c]) / std[c];
            }
        }
    }
}
 
void postProcess(const float *hm, const float *wh, const float *ang, const float *reg,
                vector<Detection> &result, const int w, const int h, const int classes,
                const int kernel_size,const float visthresh)
{
    int flag = 0;
    for (int idx = 0; idx < w * h * classes; idx++)
    {
        if (idx >= w * h * classes)
            return;
        int padding = (kernel_size - 1) / 2; //1
        int offset = -padding;               //-1
        int stride = w * h;                  //128*128
        int grid_x = idx % w;                // 纵轴坐标，x
        int grid_y = (idx / w) % h;          // 横轴坐标,y
        int cls = idx / w / h;               // 第几类
        int l, m;
        int reg_index = idx - cls * stride; // 一张图里面的位置，[0,128*128]
        float c_x, c_y;
        float objProb = Logist(hm[idx]);
        flag += 1;
 
        if (objProb > visthresh)
        {
            float max = -1;
            int max_index = 0;
            for (l = 0; l < kernel_size; ++l)
                for (m = 0; m < kernel_size; ++m)
                {
                    int cur_x = offset + l + grid_x;
                    int cur_y = offset + m + grid_y;
                    int cur_index = cur_y * w + cur_x + stride * cls;
                    int valid = (cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h);
                    float val = (valid != 0) ? Logist(hm[cur_index]) : -1;
                    max_index = (val > max) ? cur_index : max_index;
                    max = (val > max) ? val : max;
                }
 
            if (idx == max_index)
            {
                Detection det;
                c_x = grid_x + reg[reg_index];
                c_y = grid_y + reg[reg_index + stride];
                float angle = ang[reg_index];
 
                det.bbox.x1 = (c_x - wh[reg_index] / 2) * 4;
                det.bbox.y1 = (c_y - wh[reg_index + stride] / 2) * 4;
                det.bbox.x2 = (c_x + wh[reg_index] / 2) * 4;
                det.bbox.y2 = (c_y + wh[reg_index + stride] / 2) * 4;
                det.bbox.ang = angle;
                det.classId = cls;
                det.prob = objProb;
                result.push_back(det);
            }
        }
    }
}
 
void resultCorrect(std::vector<Detection> &result, const cv::Mat &img)
{
    int input_w = 512;
    int input_h = 512;
    float scale = min(float(input_w) / img.cols, float(input_h) / img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
    for (auto &item : result)
    {
        float x1 = (item.bbox.x1 - dx) / scale;
        float y1 = (item.bbox.y1 - dy) / scale;
        float x2 = (item.bbox.x2 - dx) / scale;
        float y2 = (item.bbox.y2 - dy) / scale;
        x1 = (x1 > 0) ? x1 : 0;
        y1 = (y1 > 0) ? y1 : 0;
        x2 = (x2 < img.cols) ? x2 : img.cols - 1;
        y2 = (y2 < img.rows) ? y2 : img.rows - 1;
        item.bbox.x1 = x1;
        item.bbox.y1 = y1;
        item.bbox.x2 = x2;
        item.bbox.y2 = y2;
    }
}

void draw(const std::vector<Detection> &result, cv::Mat &img)
{
    for (const auto &item : result)
    {
        float ang = item.bbox.ang;
        float cx = (item.bbox.x1 + item.bbox.x2) / 2;
        float cy = (item.bbox.y1 + item.bbox.y2) / 2;
        float height = (item.bbox.x2 - item.bbox.x1);
        float width = (item.bbox.y2 - item.bbox.y1);
        float anglePi = ang / 180 * PI;
        anglePi = anglePi < PI ? anglePi : anglePi - PI;
        float cosA = cos(anglePi);
        float sinA = sin(anglePi);
        float x1 = cx - 0.5 * width;
        float y1 = cy - 0.5 * height;
 
        float x0 = cx + 0.5 * width;
        float y0 = y1;
 
        float x2 = x1;
        float y2 = cy + 0.5 * height;
 
        float x3 = x0;
        float y3 = y2;
 
        int x0n = floor((x0 - cx) * cosA - (y0 - cy) * sinA + cx);
        int y0n = floor((x0 - cx) * sinA + (y0 - cy) * cosA + cy);
 
        int x1n = floor((x1 - cx) * cosA - (y1 - cy) * sinA + cx);
        int y1n = floor((x1 - cx) * sinA + (y1 - cy) * cosA + cy);
 
        int x2n = floor((x2 - cx) * cosA - (y2 - cy) * sinA + cx);
        int y2n = floor((x2 - cx) * sinA + (y2 - cy) * cosA + cy);
 
        int x3n = floor((x3 - cx) * cosA - (y3 - cy) * sinA + cx);
        int y3n = floor((x3 - cx) * sinA + (y3 - cy) * cosA + cy);
 
        cv::line(img, cv::Point(x0n, y0n), cv::Point(x1n, y1n), cv::Scalar(0, 0, 255), 3, 8, 0);
        cv::line(img, cv::Point(x1n, y1n), cv::Point(x2n, y2n), cv::Scalar(255, 0, 0), 3, 8, 0);
        cv::line(img, cv::Point(x2n, y2n), cv::Point(x3n, y3n), cv::Scalar(0, 0, 255), 3, 8, 0);
        cv::line(img, cv::Point(x3n, y3n), cv::Point(x0n, y0n), cv::Scalar(255, 0, 0), 3, 8, 0);
    }
}
 
int main(int argc, const char** argv)
{
    if (argc !=3)
	{
		std::cout << "you should input: \n./predict your_model_path//your_model.onnx your_img_path//your_img.jpg " << std::endl;
		return -1;
	}
	std::string model_path = argv[1];
	std::string image_file = argv[2];
    // 1. 加载模型
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "R-CenterNet"};
    Ort::SessionOptions session_option;
    session_option.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session_{env, model_path.c_str(), session_option};

    // 2. 定义输入输出层，用netron看，我这里是0，512、515、518、521，分别对应输入图片，输出hm、wh、ang、reg
    std::vector<const char *> input_names = {"0"};
    const char *const output_names[] = {"512", "515", "518", "521"};

    // 3. 加载准备推理的图片
    Mat img = imread(image_file);

    // 4. 这里是为加载的图片准备一个输入的tensor，其实只要修改512，512就行
    Ort::Value input_tensor_{nullptr};
    std::array<float, 1 * 3 * 512 * 512> input_image_{};
    std::array<int64_t, 4> input_shape_{1, 3, 512, 512};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    
    float *input_float = input_image_.data();
    fill(input_image_.begin(), input_image_.end(), 0.f);

    // 5. 预处理是把输入图片尺寸resize到网络接受的尺寸，并归一化
    preProcess(img, input_float);

    // 6. 开始推理
    std::vector<Ort::Value> ort_outputs = session_.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                       &input_tensor_, 1, output_names, 4);
 
    // 7. 后处理，提取出目标坐标、长宽、角度以及修正值，ort_outputs[0:4] -> hm wh ang reg，
    vector<Detection> result;
    postProcess(ort_outputs[0].GetTensorMutableData<float>(), ort_outputs[1].GetTensorMutableData<float>(),
                 ort_outputs[2].GetTensorMutableData<float>(), ort_outputs[3].GetTensorMutableData<float>(),
                 result, 128, 128, 1, 3, 0.3);

    // 8. 修正坐标，将坐标为负数的归为0，大于图像尺寸的置为图像边界尺寸
    resultCorrect(result, img);

    // 9. 画图
    draw(result, img);
    imwrite("result.jpg", img);
    return 0;
}