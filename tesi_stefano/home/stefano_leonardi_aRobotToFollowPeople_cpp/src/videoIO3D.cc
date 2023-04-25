#include "videoIO3D.hh"

using namespace std::chrono_literals;

///////////////////////////   LoadVideo3D Class   //////////////////////////////////////////////
LoadVideo3D::LoadVideo3D(int warmUp) : \
    warmUp(warmUp) {

    // setup streams
    init3Dcamera();
    sleepMil(warmUp);
    
    // start counting the FPS rate
    this->frameProcessed = 0;

    this->timeStart = -1; // time of start processing time
}

LoadVideo3D::~LoadVideo3D() {
  bg_thread->stop();
}

int LoadVideo3D::sourceFPS(){
    return rgb_stream->fps();
}

FrameData3D LoadVideo3D::read(){
  FrameData3D res;
    
    // normal workflow of the code
    {
      std::unique_lock<std::mutex> lock(dataMtx);
      dataCV.wait_for(lock, 200ms, [&]() { return !data.frame.empty(); });
      res = data;
      data.frame.release();
    }
    if (!res.frame.empty())
	this->frameProcessed++;
    return res;
}

double LoadVideo3D::fps(){
    double fps = this->frameProcessed / (elapsed() / 1000.0);
    // cout << fps << " = " << frameProcessed << " / (" << elapsed() << " / 1000.0)\n\n";
    return fps;
}

long long int LoadVideo3D::elapsed(){
    // the first execution will set the start time
    if(this->timeStart == -1){
        this->timeStart = timeNow();
    }

    // measure the time gone since first execution
    long long int now = timeNow();
    long long int elapsed = (now - this->timeStart + 1);   //millisecond passed from the first frame captured
    return elapsed;
}


void LoadVideo3D::init3Dcamera() {
  ctx = std::make_unique<context>();
  pipe = std::make_unique<pipeline>(*ctx);
  cfg = std::make_unique<config>();

  const int WIDTH = 640;
  const int HEIGHT = 480;
  const int FPS = 30;

  // const int WIDTH = 1280;
  // const int HEIGHT = 720;
  // const int FPS = 6;

  cfg->enable_stream(RS2_STREAM_DEPTH, 0, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);
  cfg->enable_stream(RS2_STREAM_COLOR, 0, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
  
  auto dev = cfg->resolve(*pipe).get_device();
  auto sensor = dev.query_sensors()[0];
  sensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET, rs2_rs400_visual_preset::RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY);

  pipe->start(*cfg);

  rgb_stream = std::make_unique<rs2::stream_profile>(pipe->get_active_profile().get_stream(RS2_STREAM_COLOR));
  depth_stream = std::make_unique<rs2::stream_profile>(pipe->get_active_profile().get_stream(RS2_STREAM_DEPTH));
  rgb_intrinsics = std::make_unique<rs2_intrinsics>(rgb_stream->as<rs2::video_stream_profile>().get_intrinsics());
  depth_intrinsics = std::make_unique<rs2_intrinsics>(depth_stream->as<rs2::video_stream_profile>().get_intrinsics());

  aligner = std::make_unique<rs2::align>(RS2_STREAM_COLOR);

  bg_thread = std::make_unique<single_thread>();
  bg_thread->setThreadFunction([&](const bool& terminating) {
    while (!terminating) {
      rs2::frameset frames = pipe->wait_for_frames();
      auto aligned_frames = aligner->process(frames);
      rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
      rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

      int w = color_frame.get_width();
      int h = color_frame.get_height();
      Mat img(cv::Size(w, h), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
  
      { 
        std::unique_lock<std::mutex> lock(dataMtx);
        data.frame = img;
        data.depth_frame = std::make_unique<rs2::depth_frame>(depth_frame);
        
        long long int lastTime = data.lastTime;
        long long int newTime = timeNow();
        data.lastTime = newTime;

        lock.unlock();
        dataCV.notify_one();
      
        // if (lastTime>-1) {
        //   std::cerr << "DT: " << (newTime-lastTime) << std::endl;
        // }
      }
    } 

    pipe->stop();
  }); 
  bg_thread->start();
}

bool LoadVideo3D::worldCoordinates(FrameData3D const & fd, int i, int j, float& x, float& y, float& z) {
  try {
    float depth = fd.depth_frame->get_distance(i, j);
    if (depth == 0) return false;
    float res[3];
    float point[2] = {i, j};
    rs2_deproject_pixel_to_point(res, rgb_intrinsics.get(), point, depth);
    x = res[0];
    y = res[1];
    z = res[2];
  } catch (std::exception ex) {
    std::cerr << ex.what() << std::endl;
    return false;
  }
  return true;
}
