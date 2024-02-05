#include "../include/combinator3000_api.h"

node<double> *create_node(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new node<double>(inp, outp, blocsize, samplerate);
}

channel_adapter<double> *create_channel_adapter(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new channel_adapter<double>(inp, outp, blocsize, samplerate);
}

mixer<double> *create_mixer(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new mixer<double>(inp, outp, blocsize, samplerate);
}

upsampler<double> *create_upsampler(size_t inp, size_t outp, size_t blocsize, size_t samplerate, 
    size_t num_cascade, size_t order, size_t steep)
{
    return new upsampler<double>(inp, outp, blocsize, samplerate, num_cascade, order, steep);    
}

downsampler<double> *create_downsampler(size_t inp, size_t outp, size_t blocsize, size_t samplerate, 
    size_t num_cascade, size_t order, size_t steep)
{
    return new downsampler<double>(inp, outp, blocsize, samplerate, num_cascade, order, steep);    
}

graph<double> *create_graph()
{
    return new graph<double>();
}

#ifdef FFT_NODE
#include "fft/fft_node.h"
node<double> *create_fft_node(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new fft_node<double>(inp, outp, blocsize, samplerate);
}

node<double> *create_ifft_node(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new ifft_node<double>(inp, outp, blocsize, samplerate);
}
#else
node<double> *create_fft_node(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    throw std::runtime_error("FFT node is not enabled. Recompile to enable it.");
    return new node<double>(inp, outp, blocsize, samplerate);
}

node<double> *create_ifft_node(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    throw std::runtime_error("FFT node is not enabled. Recompile to enable it.");
    return new node<double>(inp, outp, blocsize, samplerate);
}
#endif


#ifdef CSOUND_NODE
#include "csound/csound_node.h"
node<double> *create_csound_node(const char *csd_str, size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return new csound_node<double>(std::string(csd_str), inp, outp, blocsize, samplerate);
}
node<double> *create_csound_node_from_file(const char *csd_path, size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    return csound_node<double>::from_file(std::string(csd_path), inp, outp, blocsize, samplerate);
}
#else 
node<double> *create_csound_node(const char *csd_str, size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    throw std::runtime_error("Csound node is not enabled");
    return new node<double>(inp, outp, blocsize, samplerate);
}
node<double> *create_csound_node_from_file(const char *csd_path, size_t inp, size_t outp, size_t blocsize, size_t samplerate)
{
    throw std::runtime_error("Csound node is not enabled");
    return new node<double>(inp, outp, blocsize, samplerate);
}
#endif

#ifdef FAUST_JIT_NODE
#include "faust/faust_jit_node.h"
void *create_faust_jit_factory_from_file(const char *path)
{
    return (void *)faust_jit_factory<double>::from_file(path);
}
void *create_faust_jit_factory_from_string(const char *dsp_str)
{
    return (void *)faust_jit_factory<double>::from_string(dsp_str);
}

void delete_faust_jit_factory(void *f)
{
    delete (faust_jit_factory<double> *)f;
}

node<double> *create_faust_jit_node(void *factory, size_t blocsize, size_t samplerate)
{
    return new faust_jit_node<double>( (faust_jit_factory<double> *)factory, blocsize, samplerate );
}
#else
void *create_faust_jit_factory_from_file(const char *path)
{
    throw std::runtime_error("Faust JIT node is not enabled");
    return (void *)nullptr;
}
void *create_faust_jit_factory_from_string(const char *dsp_str)
{
    throw std::runtime_error("Faust JIT node is not enabled");
    return (void *)nullptr;
}

void delete_faust_jit_factory(void *f)
{
    throw std::runtime_error("Faust JIT node is not enabled");
}

node<double> *create_faust_jit_node(void *factory, size_t blocsize, size_t samplerate)
{
    throw std::runtime_error("Faust JIT node is not enabled");
    return new node<double>( (1, 1, blocsize, samplerate );
}
#endif

void delete_node(node<double> *n)
{
    delete n;
}

void delete_graph(graph<double> *g)
{
    delete g;
}

bool node_connect(node<double> *a, node<double> *b)
{
    return a->connect(b);
}

bool node_disconnect(node<double> *a, node<double> *b)
{
    return a->disconnect(b);
}

void node_process(node<double> *n, node<double> *previous)
{
    n->process(previous);
}

void graph_add_node(graph<double> *g, node<double> *n)
{
    g->add_node(n);
}

void graph_process_bloc(graph<double> *g)
{
    g->process_bloc();
}
