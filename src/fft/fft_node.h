#ifndef FFT_NODE_H
#define FFT_NODE_H

#include "combinator3000.h"
#include "AudioFFT/AudioFFT.h"

template<typename Flt>
struct fft_node : public node<Flt>
{
    // Number of output channels to allow interleaved real/imaginaries data
    // blocsize is complexsize (find input size with previous->bloc_size or with `(bloc_size-1)*2` )
    fft_node(size_t inp = 0, size_t outp = 0, size_t blocsize = 128, size_t samplerate = 48000)
        : node<Flt>::node{inp, outp, blocsize, samplerate}
        , ffts(inp)
    {
        size_t complex_size = audiofft::AudioFFT::ComplexSize(this->bloc_size);
        this->outputs = new Flt*[this->n_outputs*2]; // For storing complex numbers
        for(size_t i = 0; i < this->n_outputs*2; ++i)
            this->outputs[i] = new Flt[complex_size];
            
        for(auto & it : ffts)
            it.init(this->bloc_size);
    }

    void process(node<Flt> *previous) override 
    {
        for(size_t i = 0; i < this->n_inputs; ++i)
        {
            Flt *real = this->outputs[i*2];
            Flt *imag = this->outputs[i*2+1];
            const Flt *data = previous->outputs[i];

            ffts[i].fft(data, real, imag);
        }
    }

    std::vector<audiofft::AudioFFT> ffts;
};


template<typename Flt>
struct ifft_node : public node<Flt>
{
    ifft_node(size_t inp = 0, size_t outp = 0, size_t blocsize = 128, size_t samplerate = 48000)
        : node<Flt>::node(inp, outp, blocsize, samplerate)
        , ffts(inp)
    {
        for(auto & it : ffts)
            it.init(this->bloc_size);
    }

    void process(node<Flt> *previous) override 
    {
        for(size_t i = 0; i < this->n_outputs; ++i)
        {
            const Flt *real = previous->outputs[i*2];
            const Flt *imag = previous->outputs[i*2+1];
            ffts[i].ifft(this->outputs[i], real, imag);
        }
    }

    std::vector<audiofft::AudioFFT> ffts;
};

//template class fft_node<double>;
//template class ifft_node<double>;
#endif