#include<iostream>
#include<thread>

#include "combinator3000.h"
#include"faust/faust_node.h"
#include "faust/faust_jit_node.h"
#include"faust/tests/osc.hpp"
#include"faust/tests/filter.hpp"
#include"faust/tests/square.hpp"
#include "faust/tests/fftdel.hpp"
#include "faust/tests/fftfreeze.hpp"
#include "faust/tests/fftfilter.hpp"
#include "csound/csound_node.h"
#include "fft/fft_node.h"
#include "sndfile/sndfile_node.h"
#include "../../include/combinator3000_api.h"

#include "sndfile.hh"

void simple_test()
{
    faust_node<osc, double> *o = new faust_node<osc, double>(1024);
    faust_node<filter, double> *f = new faust_node<filter, double>(1024);
    o->connect(f);

    graph<double> g;
    g.add_node(o);
    g.process_bloc();
}

// Issue : 
// The mixer childs are executed twice, since mixer is considered twice a caller
void mix_test()
{
    SndfileHandle outfile("/home/johann/Documents/tmp/sawsquare_mixer_filt.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_32, 1, 48000 );

    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    faust_node<square, double> *o2 = new faust_node<square, double>();
    mixer<double> *m = new mixer<double>(1, 1);
    faust_node<filter, double> *f = new faust_node<filter, double>();

    // First mix signals 
    o1->connect(m);
    o2->connect(m);
    // Then send mixer output to filter
    m->connect(f);

    graph<double> g;
    g.add_node(o1);
    g.add_node(o2);

    // Buffer of 128 
    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / dur;
    for(size_t i = 0; i < npasses; ++i) 
    {
        std::cout << "npasses : " << i << " / " << npasses << std::endl;
        g.process_bloc();
        outfile.writef(f->outputs[0], 128);
    }
    
}

void resampler_test()
{
    SndfileHandle tfile("/home/johann/Documents/tmp/temoin.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_32, 1, 48000 );
    SndfileHandle upfile("/home/johann/Documents/tmp/combinator_upsample.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_32, 1, 96000 );
    SndfileHandle downfile("/home/johann/Documents/tmp/combinator_downsample.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_32, 1, 48000 );

    faust_node<osc, double> *o = new faust_node<osc, double>(128, 48000);
    upsampler<double> *up = new upsampler<double>(1, 1, 256, 96000, 1, 10, 1);
    faust_node<filter, double> *f = new faust_node<filter, double>(256, 96000);
    downsampler<double> *down = new downsampler<double>(1, 1, 128, 48000, 1, 10, 1); 

    o->connect(up);
    up->connect(f);
    f->connect(down);

    graph<double> g;
    g.add_node(o);

    // Buffer of 128 
    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / 128;


    for(size_t i = 0; i < npasses; ++i) 
    {
        g.process_bloc();
        upfile.writef(f->outputs[0], 256);
        downfile.writef(down->outputs[0], 128);
        tfile.writef(o->outputs[0], 128);
    }
}

#include "rtaudio/RtAudio.h"
#include<vector>
#include<thread>
#include<chrono>

graph<double> *g_ptr;
faust_node<filter, double> *f_ptr;

int rt_callback( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
         double streamTime, RtAudioStreamStatus status, void *userData )
{
    std::cout << "callback " << std::endl;
    g_ptr->process_bloc();
    std::cout << "block processed " << std::endl;
    ::memcpy((double*)outputBuffer, f_ptr->outputs[0], sizeof(double) * f_ptr->bloc_size);
    return 0;
}

void rt_test()
{
    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    faust_node<square, double> *o2 = new faust_node<square, double>();
    mixer<double> *m = new mixer<double>(1, 1);
    faust_node<filter, double> *f = new faust_node<filter, double>();
    graph<double> g;
    o1->connect(m);
    o2->connect(m);
    m->connect(f);
    g.add_node(o1);
    g.add_node(o2);

    std::cout << "nodes created " << std::endl;
    f_ptr = f;
    g_ptr = &g;

    RtAudio dac;
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = 1;
    parameters.firstChannel = 0;
    unsigned int sampleRate = 48000;
    unsigned int bufferFrames = 128; 

    std::cout << "RtAudio instanciated, params created " << std::endl;

    RtAudio::StreamOptions opts;
    opts.flags = RTAUDIO_NONINTERLEAVED;
    dac.openStream( &parameters, NULL, RTAUDIO_FLOAT64, sampleRate,
                        &bufferFrames, &rt_callback, (void *)&g , &opts);
    std::cout << "Stream opened " << std::endl;

    dac.startStream();
    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void simple_fft_test()
{
    SndfileHandle wf("/home/johann/Documents/tmp/fft.wav", SFM_WRITE, SF_FORMAT_WAV |SF_FORMAT_PCM_24, 1, 48000);

    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    fft_node<double> *_fft = new fft_node<double>(1, 2, 128, 48000);
    ifft_node<double> *_ifft = new ifft_node<double>(2, 1, 128, 48000);

    std::cout << "Nodes init OK " << std::endl;

    o1->connect(_fft);
    _fft->connect(_ifft);

    std::cout << "Nodes connected " << std::endl;

    graph<double> g;
    g.add_node(o1);

    std::cout << "Graph init OK " << std::endl;

    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / 128;
    for(size_t i = 0; i < npasses; ++i)
    {
        g.process_bloc();
        wf.writef(_ifft->outputs[0], 128);
    }
}

void fft_denoise_test()
{
    size_t up_bloc = 512;
    size_t up_sr = 192000;
    size_t up_factor = 2;

    faust_node<osc, double> *o1 = new faust_node<osc, double>(128, 48000);
    upsampler<double> *up = new upsampler<double>(1, 1, up_bloc, up_sr, up_factor, 10, 1);

    // FFT 
    fft_node<double> *_fft = new fft_node<double>(1, 3, up_bloc, up_sr);
    faust_node<fftfreeze, double> *_delfft = new faust_node<fftfreeze, double>(up_bloc/2-1, up_sr);
    ifft_node<double> *_ifft = new ifft_node<double>(3, 1, up_bloc, up_sr);

    // Downsample
    downsampler<double> *down = new downsampler<double>(1, 1, 128, 48000, up_factor, 10, 1);

    sndwrite_node<double> *sndw = new sndwrite_node<double>("/home/johann/Documents/tmp/fft_del.wav", 1, 128, 48000);
    
    _delfft->setParamValue("fftSize", 512);
    //_delfft->setParamValue("freezeBtn", 1);

    o1->connect(up);
    up->connect(_fft);
    _fft->connect(_ifft);
    _delfft->connect(_ifft);
    _ifft->connect(down);
    down->connect(sndw);

    std::cout << "!!! Print after connecting, before adding to node " << std::endl;
    auto ff = [](node<double> *ptr) {
        std::cout << ptr->get_name() << std::endl;
        for(auto & it : ptr->connections)
            std::cout << "\tconnections - " << it.target->get_name() << " & chans  " << it.output_range.second << std::endl; 
    };

    ff(o1);
    ff(up);
    ff(_fft);
    ff(_ifft);

    /*
    up->connect(_fft);
    _fft->connect(_delfft);
    _delfft->connect(_ifft);
    _ifft->connect(down);
    */

    rtgraph<double> g(0, 1, 128, 48000);
    g.add_node(o1);

    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / 128;

    std::cout << g.generate_patchbook_code() << std::endl;

    g.start_stream();
    /*
    for(size_t i = 0; i < npasses; ++i)
    {
        if(i < npasses/2 )
        {
            _delfft->setParamValue("freezeBtn", 1);
        } else 
            _delfft->setParamValue("freezeBtn", 0);
        g.process_bloc();
    }
    */

    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}


void faust_jit_test()
{
    SndfileHandle wf("/home/johann/Documents/tmp/faust_jit.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_24, 1, 48000);

    faust_jit_factory<double> *fac = faust_jit_factory<double>::from_string("import(\"stdfaust.lib\"); process = os.sawtooth(200) * 0.2;", "sawtooth");
    faust_jit_node<double> *dsp = new faust_jit_node<double>(fac, 128, 48000);
    faust_node<filter, double> *filt = new faust_node<filter, double>(128, 48000);

    dsp->connect(filt);

    graph<double> g;
    g.add_node(dsp);

    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / 128;
    for(size_t i = 0; i < npasses; ++i)
    {
        g.process_bloc();
        wf.writef(filt->outputs[0], 128);
    }
}

void csound_faust_test()
{
    faust_jit_factory<double> *fac = faust_jit_factory<double>::from_string("import(\"stdfaust.lib\"); freq = hslider(\"freq\", 100, 50, 1000, 0.1); process = os.sawtooth(freq) * 0.2;", "sawtooth");
    faust_jit_node<double> *fdsp = new faust_jit_node<double>(fac, 128, 48000);
    fdsp->setParamValue("freq", 300);
    std::string csd = "" \
    "<CsoundSynthesizer>\n" \
    "<CsOptions>\n" \
    "</CsOptions>\n" \
    "<CsInstruments> \n" \
    "instr 1 \n" \
        "ain = inch(1) \n" \
        "adel = abs(oscili:a(0.02, 0.5)) \n" \
        "kfb  = 0.7 \n" \
        "ao = flanger(ain, adel, kfb) \n" \
        "outch 1, ao \n" \
    "endin \n" \
    "</CsInstruments> \n" \
    "<CsScore> \n" \
        "f 0 z \n" \
        "i 1 0 -1 \n" \
    "</CsScore> \n" \
    "</CsoundSynthesizer> \n";
    std::cout << csd << std::endl;
    csound_node<double> *csn = new csound_node<double>(csd, 1, 1, 128, 48000);
    fdsp->setParamValue("freq", 100);

    fdsp->connect(csn);

    graph<> g;
    g.add_node(fdsp);

    SndfileHandle wf("/home/johann/Documents/tmp/csound_faust_jit.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_24, 1, 48000);
    size_t dur = 10; // seconds 
    size_t nsamps_total = dur * 48000;
    size_t npasses = nsamps_total / 128;
    for(size_t i = 0; i < npasses; ++i)
    {
        g.process_bloc();
        wf.writef(csn->outputs[0], 128);
    }
}


/*
    Test construct destruct
*/
void test_api()
{
    upsampler<double> *u = create_upsampler(1, 1, 512, 192000, 2, 10, 1);

    std::cout << "upsampler ptr : " << u << std::endl;
    delete_node((node<double> *) u);
}

void test_rtgraph()
{
    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    print("Osc 1 : " + std::to_string(intptr_t(o1)));
    faust_node<square, double> *o2 = new faust_node<square, double>();
    print("Osc 2 : " + std::to_string(intptr_t(o2)));
    mixer<double> *m = new mixer<double>(1, 1);
    print("Mixer : " + std::to_string(intptr_t(m)));
    faust_node<filter, double> *f = new faust_node<filter, double>();
    print("Filter 1 : " + std::to_string(intptr_t(f)));

    o1->set_name("Oscillator1");
    o2->set_name("Oscillator2");
    m->set_name("OscillatorMixer");
    f->set_name("Filter");

    std::cout << "Nodes created " << std::endl;

    o1->connect(m);
    o2->connect(m);
    m->connect(f);

    std::cout << "Nodes connected " << std::endl;

    rtgraph<double> g(0, 1, 128, 48000);

    std::cout << "Graph created " << std::endl;

    g.add_node(o1);
    g.add_node(o2);

    std::cout << "Nodes added, starting perf " << std::endl;

    g.start_stream();


    std::string code = g.generate_patchbook_code();
    std::cout << code << std::endl;

    while(true) 
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void test_complex_graph()
{
    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    o1->set_name("Oscillator 1");
    faust_node<square, double> *o2 = new faust_node<square, double>();
    o2->set_name("Oscillator 2");
    faust_node<square, double> *o3 = new faust_node<square, double>();
    o3->set_name("Oscillator 3");
    mixer<double> *m = new mixer<double>(1, 1);
    m->set_name("Mixer");
    faust_node<filter, double> *f1 = new faust_node<filter, double>();
    f1->set_name("Filter 1");
    faust_node<filter, double> *f2 = new faust_node<filter, double>();
    f2->set_name("Filter 2");

    o1->connect(m);
    o2->connect(m);
    o3->connect(f2);
    m->connect(f1);

    graph<double> g(0, 1);
    g.add_node(o1);
    g.add_node(o2);
    g.add_node(o3);

    std::string code = g.generate_patchbook_code();
    std::cout << code << std::endl;

}

void test_complex_graph2()
{
    node<double> *o1 = new node<double>(0, 4);
    o1->set_name("Osc1");
    node<double> *o2 = new node<double>(0, 2);
    o2->set_name("Osc2");
    node<double> *o3 = new node<double>(0, 1);
    o3->set_name("Osc3");

    node<double> *f1 = new node<double>(4, 1);
    f1->set_name("filt1");
    node<double> *f2 = new node<double>(2, 2);
    f2->set_name("filt2");

    o1->connect(f2);
    o2->connect(f2);
    o3->connect(f1);

    graph<double> g(0, 2);
    g.add_node(o1);
    g.add_node(o2);
    g.add_node(o3);

    std::string code = g.generate_patchbook_code();
    std::cout << code << std::endl;
}

void test_sndwrite()
{
    faust_node<osc, double> *o1 = new faust_node<osc, double>();
    faust_node<square, double> *o2 = new faust_node<square, double>();
    mixer<double> *m = new mixer<double>(1, 1);
    faust_node<filter, double> *f = new faust_node<filter, double>();
    sndwrite_node<double> *w = new sndwrite_node<double>("/home/johann/Documents/tmp/sndwrite.wav", 1, 128, 48000);

    o1->connect(m);
    o2->connect(m);
    m->connect(f);
    f->connect(w);

    rtgraph<double> g(0, 1, 128, 48000);

    g.add_node(o1);
    g.add_node(o2);

    g.start_stream();

    std::string code = g.generate_patchbook_code();
    std::cout << code << std::endl;
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void test_sndread()
{
    sndread_node<double> *s = new sndread_node<double>("/home/johann/Documents/tmp/Tears of exhaustion.wav", 128, 48000 );
    faust_node<filter, double> *f = new faust_node<filter, double> (128, 48000);

    s->connect(f);

    std::cout << "outputs : " << s->n_outputs << std::endl;
    rtgraph<double> g(0, 1, 128, 48000);

    g.add_node(s);
    g.start_stream();

    std::cout << g.generate_patchbook_code() << std::endl;
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void test_sndread_stereo()
{

    sndread_node<double> *s = new sndread_node<double>("/home/johann/Documents/tmp/noisy.wav", 128, 48000 );
    node<double> *n = new node<double>(2, 2, 128, 48000);

    std::cout << "outputs : " << s->n_outputs << std::endl;
    rtgraph<double> g(0, 2, 128, 48000);

    g.add_node(s);
    g.start_stream();

    std::cout << g.generate_patchbook_code() << std::endl;
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void test_single()
{
    sndread_node<double> *s = new sndread_node<double>("/home/johann/Documents/tmp/noisy.wav", 2048, 48000 );

    std::cout << "outputs : " << s->n_outputs << std::endl;
    rtgraph<double> g(0, 2, 2048, 48000);

    g.add_node(s);
    g.start_stream();

    std::cout << g.generate_patchbook_code() << std::endl;
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// Works fine with a bit of glitch. Needs overlap add.
void fft_denoiser_test()
{

    //faust_node<osc, double> *in = new faust_node<osc, double> (2048, 48000);
    sndread_node<double> *sndin = new sndread_node<double>("/home/johann/Documents/tmp/tears.wav", 2048, 48000);
    fft_node<double> *fft = new fft_node<double>(1, 3, 2048, 48000);
    faust_node<fftfilter, double> *filt = new faust_node<fftfilter, double>(1024, 48000);
    ifft_node<double> *ifft = new ifft_node<double>(3, 1, 2048, 48000);

    sndin->connect({fft, {0, 0}, 0});
    //fft->connect(ifft);
    fft->connect(filt);
    filt->connect(ifft);

    filt->setParamValue("fftSize", 2048);
    filt->setParamValue("cut", 150);
    filt->setParamValue("gain", 0.0);
    

    rtgraph<double> g(0, 1, 2048, 48000);

    g.add_node(sndin);
    std::cout << g.generate_patchbook_code() << std::endl;
    g.generate_faust_diagram();
    g.start_stream();

    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void fft_gain_test()
{

    //faust_node<osc, double> *in = new faust_node<osc, double> (2048, 48000);
    sndread_node<double> *sndin = new sndread_node<double>("/home/johann/Documents/tmp/tears.wav", 2048, 48000);
    fft_node<double> *fft = new fft_node<double>(1, 3, 2048, 48000);
    faust_node<fftfilter, double> *filt = new faust_node<fftfilter, double>(1023, 48000);
    ifft_node<double> *ifft = new ifft_node<double>(3, 1, 2048, 48000);

    sndin->connect(fft);
    fft->connect(filt);
    filt->connect(ifft);

    filt->setParamValue("fftSize", 2048);
    filt->setParamValue("cut", 1000);
    filt->setParamValue("gain", 0.7);
    
    rtgraph<double> g(0, 1, 2048, 48000);

    g.add_node(sndin);
    std::cout << g.generate_patchbook_code() << std::endl;
    g.generate_faust_diagram();
    g.start_stream();

    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

/*
    Increasing and decreasing the bloc size 
*/

void test_upbloc()
{
    sndread_node<double> *sndin = new sndread_node<double>("/home/johann/Documents/tmp/tears.wav", 128, 48000);
    upbloc<double> *up = new upbloc<double>(1, 1, 2048, 48000);
    downbloc<double> *down = new downbloc<double>(1, 1, 128, 48000);

    sndin->connect(up);
    up->connect(down);

    rtgraph<double> g(0, 1, 128, 48000);
    g.add_node(sndin);

    std::cout << g.generate_patchbook_code() << std::endl;

    g.start_stream();

    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}


// Overlap add
void test_overlap_fft()
{
    faust_node<osc, double> *sndin = new faust_node<osc, double> (1024, 48000);
    upbloc<double> *up = new upbloc<double>(1, 1, 4096, 48000);
    over_fft<double> *fft = new over_fft<double>(1, 3, 4096, 48000, 2);
    faust_node<fftfilter, double> *filt = new faust_node<fftfilter, double>(4096, 48000);
    over_ifft<double> *ifft = new over_ifft<double>(3, 1, 4096, 48000, 2);

    // Only returns buffers of 2048 
    over_ifft_downbloc<double> *ifftdown = new over_ifft_downbloc<double>(1, 1, 4096, 48000, 2);
    downbloc<double> *down = new downbloc<double>(1, 1, 1024, 48000);
    sndwrite_node<double> *wr = new sndwrite_node<double>("/home/johann/Documents/tmp/overlap.wav", 1, 1024, 48000);

    filt->setParamValue("fftSize", 2048);
    filt->setParamValue("cut", 500);
    filt->setParamValue("gain", 0.0);

    sndin->connect(up);
    up->connect(fft);
    fft->connect(filt);
    filt->connect(ifft);
    ifft->connect(ifftdown);
    ifftdown->connect(down);
    down->connect(wr);
    //fft->connect(ifft);
    //filt->connect(ifft);
    //ifft->connect(down);

    graph<double> g(0, 1, 1024, 48000);
    g.add_node(sndin);

    std::cout << g.generate_patchbook_code() << std::endl;

    //g.start_stream();

    while(true)
    {
        g.process_bloc();
    }
}

// Real complex situation : 
// * Csound amp following on input driving a Faust synthesizer
// * FFT denoising 

int main()
{
    //simple_test();
    //mix_test();

    // Non realtime 
    //resampler_test();
    //simple_fft_test();
    //faust_jit_test();
    //csound_faust_test();

    // Then realtime 
    //fft_denoise_test(); 
    //rt_test();
    //test_api();
    //test_rtgraph();
    //test_complex_graph();
    //test_complex_graph2();
    //test_sndwrite();
    //test_sndread();
    //test_sndread_stereo();
    //test_single();
    //fft_denoiser_test();
    //fft_gain_test();
    //test_upbloc();
    test_overlap_fft();
    return 0;
}