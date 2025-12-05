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

#define MINIAUDIO_IMPLEMENTATION

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


void sig_to_csv(std::vector<double> &v, std::string path)
{
    std::string s;
    std::string sep = "";
    for(auto & it : v)
    {
        s = s + sep + std::to_string(it);
        sep = ", ";
    }

    std::ofstream ofs(path);
    ofs << s;
    ofs.close();
}

/*
http://recherche.ircam.fr/pub/dafx11/Papers/27_e.pdf
Beau papier sur l'overlap/add : https://perso.telecom-paristech.fr/gpeeters/doc/Peeters_2001_PhDThesisv1.1.pdf
Another one https://ccrma.stanford.edu/~jos/OLA/OLA_2up.pdf
https://dsp.stackexchange.com/questions/13436/choosing-the-right-overlap-for-a-window-function/33615#33615

https://holometer.fnal.gov/GH_FFT.pdf
*/

#include "sndfile.hh"
#include"fft/AudioFFT/AudioFFT.h"
void test_pure_fft()
{
    size_t bsize = 128;
    size_t fftsize = 8192;
    size_t overlap = 16;
    size_t hop_size = fftsize / overlap;
    size_t complex_size = audiofft::AudioFFT::ComplexSize(fftsize);

    SndfileHandle r("/home/johann/Documents/tmp/field.wav", SFM_READ);
    SndfileHandle w("/home/johann/Documents/tmp/tearsFFT.wav", SFM_WRITE, r.format(), r.channels(), r.samplerate());

    audio_context ctx{0, r.channels(), bsize, r.samplerate()};

    node<double> *dummy = new node<double>(0, 3, complex_size, r.samplerate());
    faust_node<fftfilter, double> * f = new faust_node<fftfilter, double>(fftsize, r.samplerate());
    connection<double> c{dummy, {0, 2}, 0};
    f->n_nodes_in = 1;
    f->setParamValue("cut", 500);
    f->setParamValue("fftSize", fftsize);
    f->setParamValue("gain", 0);

    double fq_resolution = double(r.samplerate()) / double(fftsize);

    std::vector<audiofft::AudioFFT>ffts(r.channels());
    for(auto & it : ffts)
        it.init(fftsize);

    double *read_buf = new double[bsize*r.channels()]();
    double *winbuf = new double[fftsize]();
    double **fft_bufs = new double*[r.channels()]();
    for(size_t i = 0; i < r.channels(); ++i)
        fft_bufs[i] = new double[fftsize]();


    double **overlap_bufs = new double*[r.channels() * overlap]();
    for(size_t i = 0; i < (r.channels() * overlap); ++i)
        overlap_bufs[i] = new double[fftsize]();

    double **out_bufs = new double *[r.channels()]();
    for(size_t i = 0; i < r.channels(); ++i)
        out_bufs[i] = new double[hop_size];

    double **complex_bufs = new double*[r.channels() * 3];
    for(size_t i = 0; i < r.channels(); ++i)
    {
        complex_bufs[i*3] = new double[complex_size](); 
        complex_bufs[i*3+1] = new double[complex_size](); 
        complex_bufs[i*3+2] = new double[complex_size]; 
        for(size_t n = 0; n < complex_size; ++n)
            complex_bufs[i*3+2][n] = n;
    }

    std::cout << "All init/alloc ok " << std::endl;
    
    size_t cnt = 0;
    while(r.readf(read_buf, bsize) > 0)
    {
        // deinterleave
        for(size_t ch = 0; ch < r.channels(); ++ch)
            for(size_t i = 0; i < bsize; ++i)
                fft_bufs[ch][(fftsize-hop_size)+cnt+i] = read_buf[i * r.channels() + ch];

        std::cout << "Deinterleaved " << std::endl;
        
        std::cout << "FFT Processed " << std::endl;

        // read in outbufs to fill interleaved sndfile buffer
        for(size_t ch = 0; ch < r.channels(); ++ch)
        {
            for(size_t i = 0; i < bsize; ++i)
            {
                read_buf[i*r.channels()+ch] = out_bufs[ch][i+cnt];
            }
        }

        std::cout << "Interleaved " << std::endl;
        w.writef(read_buf, bsize);
        std::cout << "wrote to snd out  " << std::endl;
        cnt = (cnt + bsize) % hop_size ;
        if(cnt == 0 ) // process fft
        {
            for(size_t ch = 0; ch < r.channels(); ++ch)
            {
                for(size_t n = 0; n < fftsize; ++n)
                {
                    winbuf[n] = fft_bufs[ch][n] * hanning(n, fftsize);
                }
                ffts[ch].fft(winbuf, complex_bufs[ch*3], complex_bufs[ch*3+1]);
                // Copy to give room for further input samples at (fftsize - hop_size)
                std::copy(fft_bufs[ch]+hop_size, fft_bufs[ch]+fftsize, fft_bufs[ch]);

                // Do something with FFT Here
                std::cout << "copy to dummy" << std::endl;
                for(size_t i = 0; i < 3; ++i)
                    std::copy(complex_bufs[ch*3+i], complex_bufs[ch*3+i]+complex_size, dummy->outputs[i]);
                std::cout << "faust process " << std::endl;
                f->process(c, ctx);
                std::cout  << "processed, recopying " << std::endl;
                for(size_t i = 0; i < 3; ++i)
                    std::copy(f->outputs[i], f->outputs[i]+complex_size, complex_bufs[ch*3+i]);
                std::cout << "all ok " << std::endl;


                // First copy all overlaps to the next one and IFFT to the first one
                // Could be done with circular buffers instead (save computation)
                for(size_t o = (overlap - 1); o > 0; o--)
                    std::copy(overlap_bufs[ch*overlap+(o-1)], overlap_bufs[ch*overlap+(o-1)] + fftsize, overlap_bufs[ch*overlap+o]);
                /*
                for(size_t o = 0; o < (overlap-1) ; ++o)
                    std::copy(overlap_bufs[ch*overlap+o], overlap_bufs[ch*overlap+o] + fftsize, overlap_bufs[ch*overlap+o+1]);
                */

                //ffts[ch].ifft(overlap_bufs[ch*overlap], complex_bufs[ch*3], complex_bufs[ch*3+1]);
                ffts[ch].ifft(overlap_bufs[ch*overlap], complex_bufs[ch*3], complex_bufs[ch*3+1]);
                //std::copy(overlap_bufs[ch*overlap], overlap_bufs[ch*overlap] + hop_size, out_bufs[ch]);
                for(size_t n = 0; n < fftsize; ++n)
                {

                    size_t mod = fftsize / (overlap/2);
                    //overlap_bufs[ch*overlap][n] *= (hanning(n%mod, mod)) ;
                    //overlap_bufs[ch*overlap][n] *= (root_hann(n%mod, mod)) ;
                    overlap_bufs[ch*overlap][n] *= hanning(n, fftsize);
                }

                // Checking overlap bufs
                std::vector<double> xdata(fftsize);
                std::vector<double> ydata(fftsize);
                for(size_t i = 0; i < fftsize; ++i)
                {
                    xdata[i] = i;
                }
                for(size_t y = 0; y < overlap; ++y)
                {
                    AsciiPlotter p(std::string("Overlap buffs channel " + std::to_string(ch)) , 90, 15);
                    for(size_t i = 0; i < fftsize; ++i)
                        ydata[i] = overlap_bufs[ch*overlap+y][i];
                    p.addPlot(xdata, ydata, std::string("overlap" + std::to_string(y)), '*');
                    p.show();
                }

                for(size_t n = 0; n < hop_size; ++n)
                {
                    double sum = 0.0;
                    // Working with overlap 2
                    /*for(size_t o = 0; o < overlap; ++o )
                    {
                        size_t pos_index = (overlap-1) - o;
                        size_t pos_offset = pos_index * hop_size;
                        //sum += overlap_bufs[ch*overlap+o][pos_offset+n];
                        
                        // This one works for overlap 2 
                        sum += overlap_bufs[ch*overlap+o][n+(o*hop_size)];
                    }
                    */

                   // This works, but only sums/computes last incoming 2 overlaps segments
                   // Needs the windowing to be done on fftsize/(overlap/2) times 
                   
                   /*size_t mod = (overlap / 2);
                   for(size_t o = 0; o < 2; ++o)
                   {   
                        sum += (overlap_bufs[ch*overlap+o][n+(o*hop_size)]);
                   }
                   sum *= overlap*mod;
                   */

                // This method works fine (with or without synthesis windowing)
                // If synthesis windowing (WOLA) > big fft size with small overlap create LFO on amplitude
                  for(size_t m = 0; m < overlap; ++m)
                  {
                    sum += overlap_bufs[ch*overlap+m][n+(m*hop_size)];
                  }
                  double nwindows = (overlap / 2);
                  sum /= (nwindows);
                   
                
                  /*for(size_t m = 0; m < overlap; ++m)
                  {
                    size_t w_index = ((overlap-m)%overlap) * hop_size;
                    sum += overlap_bufs[ch*overlap+m][n+(m*hop_size)] * hanning( (w_index + n), fftsize );
                  }
                  */

                   // This produces no glitch, but a windowing LFO
                   /*
                   for(size_t m = 0; m < overlap; ++m)
                   {
                        size_t nm = (m+2) % overlap;
                        sum += overlap_bufs[ch*overlap+m][n+(nm*hop_size)] * root_hann( (n+(m*hop_size))%hop_size, hop_size);
                   }
                   sum /= overlap;
                    */

                   // This works, but creates some time stretching and spectral stuff
                   // Also create LFO for overlap 2 (not 4)
                   /*for(size_t m = 0; m < overlap; ++m)
                   {
                        sum += overlap_bufs[ch*overlap+m][(fftsize-(m*hop_size)-hop_size)+n] * root_hann(n+(m*hop_size), fftsize);
                   }*/





                   /*
                   for(size_t m = 0; m < overlap; ++m)
                   {
                    for(size_t o = 0; o < overlap; ++o)
                    {
                        size_t idx = (o + m) % overlap;
                        sum += overlap_bufs[ch*overlap+o][n+(idx*hop_size)];
                    }
                   }
                   sum /= overlap;
                   */
                   /*
                   size_t mod = fftsize / (overlap);
                   for(size_t o = 0; o < overlap; ++o)
                   {
                        sum += overlap_bufs[ch*overlap+o][n+(o*hop_size)];
                   }
                   */

                    // Try to "combine" windows by multiplying them 

                    out_bufs[ch][n] = sum;
                }
                std::cout << "\n\n" << std::endl;
                AsciiPlotter pp("Outbuf", 90, 15);
                xdata.resize(hop_size);
                ydata.resize(hop_size);
                for(size_t i = 0; i < hop_size; ++i)
                {
                    xdata[i] = i;
                    ydata[i] = out_bufs[ch][i];
                }
                pp.addPlot(xdata, ydata, "Output", '-');
                pp.show();
            }
        }

    }
};


void test_ola_fft_nodes()
{
    size_t fft_size = 8192;
    size_t overlap = 16;
    size_t sr = 48000;
    size_t bloc_size = 64;

    sndread_node<double> *snd = new sndread_node<double>("/home/johann/Documents/tmp/tearsmono.wav", bloc_size);    
    ola_fft<double> *fft = new ola_fft<double>(1, 3, fft_size, overlap, sr);
    faust_node<fftfilter, double> *f = new faust_node<fftfilter, double>(fft_size, sr);
    ola_ifft<double> *ifft = new ola_ifft<double>(3, 1, fft_size, overlap, sr);
    //circular_downbloc<double> *down = new circular_downbloc<double>(1, 1, bloc_size, sr);
    sndwrite_node<double> *w = new sndwrite_node<double>("/home/johann/Documents/tmp/OLATest.wav", 1, bloc_size, sr);

    f->setParamValue("fftSize", fft_size);
    f->setParamValue("gain", 0.05);
    f->setParamValue("cut", 1000.0);
    
    snd->connect(fft);
    fft->connect(f);
    f->connect(ifft);
    ifft->connect(w);
    //ifft->connect(down);
    //down->connect(w);

    mini_rtgraph<double> g(0, 1, bloc_size, sr);
    g.add_node(snd);
    std::cout << g.generate_patchbook_code() << std::endl;

    g.start_stream();
    while(true)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

}

// Real complex situation : 
// * Csound amp following on input driving a Faust synthesizer
// * FFT denoising 


#include<signal.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    ucontext_t *ctx = (ucontext_t *)arg;
    ctx->uc_mcontext.gregs[REG_RIP]++;
    printf("Exception segfault at %p\n", si->si_addr);
    exit(0);
}

int main()
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);


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
    //test_nonoverlap_fft();
    //test_ola();
    

    // Simple test of working OLA FFT transform > Working 
    //test_pure_fft();

    test_ola_fft_nodes();
    return 0;
}