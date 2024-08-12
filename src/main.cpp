#include <iostream>
#include <fstream>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <NvInferRuntime.h>

#include <cassert>
#include <ctime>
#include <iomanip>

using namespace std;
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept {
        // Set log level
        if (severity <= Severity::kINFO) {
            timePrefix();
            std::cout << severityPrefix(severity) << "[TRT] " << std::string(msg) << std::endl;
        }
    }

private:
    static const char *severityPrefix(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                return "[F] ";
            case Severity::kERROR:
                return "[E] ";
            case Severity::kWARNING:
                return "[W] ";
            case Severity::kINFO:
                return "[I] ";
            case Severity::kVERBOSE:
                return "[V] ";
            default:
                assert(0);
                return "";
        }
    }

    void timePrefix() {
        std::time_t timestamp = std::time(nullptr);
        tm *tm_local = std::localtime(&timestamp);
        std::cout << "[";
        std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon
                  << "/";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
        std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year
                  << "-";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
    }
};

int main() {

    Logger logger;

    // Load the TensorRT engine file
    ifstream engineFile("/home/egorundel/projects/experiment/trt_engine_inspector/YOLOv8_m_night_INT8_bs12.trt", ios::binary);
    if (!engineFile) {
        cout << "Failed to open engine file." << endl;
        return -1;
    }

    // Read the engine file into a buffer
    engineFile.seekg(0, ios::end);
    size_t fsize = engineFile.tellg();
    engineFile.seekg(0, ios::beg);
    vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();

    // Create a runtime instance
    IRuntime* runtime = createInferRuntime(logger);
    initLibNvInferPlugins(&logger, "");
    if (!runtime) {
        cout << "Failed to create runtime." << endl;
        return -1;
    }

    // Deserialize the engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine) {
        cout << "Failed to deserialize engine." << endl;
        delete runtime;
        return -1;
    }

    // Iterate through the layers to get their precision
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        cout << "Failed to create execution context." << endl;
        delete engine;
        delete runtime;
        return -1;
    }

    IEngineInspector* inspector = engine->createEngineInspector();
    inspector->setExecutionContext(context);

    cout << inspector->getLayerInformation(0, LayerInformationFormat::kJSON); 
    cout << inspector->getEngineInformation(LayerInformationFormat::kONELINE);

    std::ofstream engine_log;
    engine_log.open("./engine_log.json",ios_base::out);
    engine_log << inspector->getEngineInformation(LayerInformationFormat::kJSON) << endl;
    engine_log.close();

    // Clean up
    delete inspector;
    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}