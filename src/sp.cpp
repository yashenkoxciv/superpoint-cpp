#include <memory>
#include <string>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bits/stdc++.h>

#include "argparse.hpp"


int main(int argc, const char** argv) {
    ArgumentParser parser;

    parser.addArgument("-i", "--input", 1, true);
    parser.addArgument("-m", "--model", 1, true);
    parser.addArgument("-d", "--device", 1, true);

    parser.parse(argc, argv);

    std::string usr_device = parser.retrieve<std::string>("device");
    std::string model_pt = parser.retrieve<std::string>("model");
    std::string input_dir = parser.retrieve<std::string>("input");

    // choose device cuda | cpu
    torch::Device device = torch::kCPU;
    if (usr_device.compare("cuda") == 0) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA" << std::endl;
        } else {
            std::cout << "CUDA is not available! Using CPU." << std::endl;
        }
    } else {
        std::cout << "Using CPU" << std::endl;
    }
    
    // load model
    std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(model_pt);
    model->to(device);

    // fake data as example
    /* 
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(
        torch::rand({1, 1, 360, 240}).to(device)
    );
    */
    std::vector<cv::String> fn;
    cv::glob(input_dir + "/*.png", fn, false);

    size_t count = fn.size();
    cv::Mat image, greyimage, rimage;
    cv::Size target_size(160, 120);
    std::vector<double> tms;
    double c_time;
    for (size_t i = 0; i < count; i++) {
        std::cout << '\r' << i + 1 << std::flush;
        image = cv::imread(fn[i]);
        cv::cvtColor(image, greyimage, CV_BGR2GRAY);
        cv::resize(greyimage, rimage, target_size);
        //cv::imwrite("output.png", rimage);

        std::vector<torch::jit::IValue> input;
        torch::Tensor x = torch::from_blob(rimage.data, {1, 1, rimage.rows, rimage.cols}, at::kByte).toType(at::kFloat).to(device) / 255;

        input.push_back(x);

        std::clock_t start, end;
        start = std::clock();
        // time to inference 
        auto z = model->forward(input).toTuple();
        auto p = z->elements()[0].toTensor();
        auto d = z->elements()[1].toTensor();

        end = std::clock();

        c_time = (double)(end - start) / CLOCKS_PER_SEC;
        tms.push_back(c_time);
    }
    std::cout << std::endl;
    double et = torch::from_blob(tms.data(), {(int)tms.size()}, at::kDouble).mean().item<double>();
    double ms = et*1000;
    double fps = 1000 / ms;
    std::cout << ms << " ms, " << fps << " fps" << std::endl;
}

