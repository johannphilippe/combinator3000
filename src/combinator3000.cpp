#include "combinator3000.h"
#include<cstring>
#include "comb_mem.h"

template<typename Flt>
node<Flt>::node( size_t inp, size_t outp, size_t blocsize, size_t samplerate, bool alloc_memory)
    : n_inputs(inp)
    , n_outputs(outp)
    , bloc_size(blocsize)
    , sample_rate(samplerate)
    , n_nodes_in(0)
{
    if(!is_power_of_two(bloc_size) && (bloc_size != 1))
        throw std::runtime_error("Node bloc size must be power of two or equal 1");
    this->set_name("Node");
    if(alloc_memory)
    {
        outputs = new Flt*[n_outputs];
        main_mem->alloc_channels<Flt>(bloc_size, n_outputs, outputs);
    }
}

template<typename Flt>
node<Flt>::~node()
{
    for(auto & it : connections)
        it.target->n_nodes_in -= 1;
    for(size_t i = 0; i < n_outputs; ++i)
        delete outputs[i];
    delete outputs;
}

template<typename Flt>
bool node<Flt>::connect(node<Flt> *n, bool adapt_channels)
{
    if(this->n_outputs == n->n_inputs) 
    {
        connections.push_back(connection<Flt>{n, {0, this->n_outputs - 1}, 0});
        n->n_nodes_in++;
        return true;
    } else 
    { 
        if(adapt_channels) 
        {
            channel_adapter<Flt> *a = 
                new channel_adapter<Flt>(this->n_outputs, n->n_inputs, 
                    this->bloc_size, this->sample_rate);
            connections.push_back({a, {0, this->n_outputs - 1}, 0});
            a->n_nodes_in++;
            a->connect(n);
            return true;
        } else 
        {
            if(this->n_outputs > n->n_inputs) 
            {
                connections.push_back({n, {0, n->n_inputs - 1}, 0 });
            } else if(this->n_outputs < this->n_inputs)
            {
                connections.push_back({n, {0, this->n_outputs - 1}, 0});
            }
            n->n_nodes_in++;
            return true;
        }
    }
    return false;
}

template<typename Flt>
bool node<Flt>::connect(connection<Flt> n, bool adapt_channels)
{
    /*
    if(n.output_range.first == 0
        && n.output_range.second == 0
        && n.input_offset == 0)
            return this->connect(n.target);
    */
    size_t num_outputs = n.output_range.second - n.output_range.first + 1;
    size_t num_inputs = n.target->n_inputs - n.input_offset;

    if(num_outputs > this->n_outputs
        || num_inputs > n.target->n_inputs)
    {
        std::cout << "connection<Flt> error - channels sizes must match node actual number of channels : " 
            << this->get_name() << " >> while trying to connect to >> " 
            << n.target->get_name() << std::endl;
        return false;
    }

    if(num_inputs >= num_outputs) 
    {
        connections.push_back(n);
        n.target->n_nodes_in++;
    } else  
        // the required connection asks more channels connected than target inputs
    {
        if(adapt_channels)  // Try to fit the number of desired outputs to the of inputs of target
        {
            channel_adapter<Flt> *a = 
                new channel_adapter<Flt>(num_outputs, num_inputs, 
                    this->bloc_size, this->sample_rate);
            this->connect({a, n.output_range, n.input_offset});
            a->n_nodes_in++;
            a->connect({n.target, {0, 0}, n.input_offset});
        } else  // Some outputs will go to nothing (over the inputs limits of target)
        {
            size_t diff = num_outputs - num_inputs;
            this->connect({n.target, {n.output_range.first, n.output_range.second - diff}
                , n.input_offset});
            n.target->n_nodes_in++;
        }
        return true;
    }
    return false;
}

template<typename Flt>
bool node<Flt>::disconnect(node<Flt> *n)
{
    for(size_t i = 0; i < connections.size(); ++i)
    {
        if(connections[i].target == n) {
            connections[i].target->n_nodes_in -= 1;
            connections.erase(connections.begin() + i);
            return true;
        }
    }
    return false;
}

template<typename Flt>
void node<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    // Copying inputs to outputs (input is the output of previous node)
    if(n_inputs > 0) 
    {
        for(size_t ch = previous.output_range.first, i = previous.input_offset; 
            ch <= previous.output_range.second; ++ch, ++i)

            std::copy(previous.target->outputs[ch], 
                previous.target->outputs[ch] + previous.target->bloc_size, 
                this->outputs[i]);
    }
}

template<typename Flt>
void node<Flt>::set_name(std::string n) {this->name = name_gen::concat(n);}

template<typename Flt>
std::string &node<Flt>::get_name() {return this->name;}

template class node<double>;
template class node<float>;

template<typename Flt>
channel_adapter<Flt>::channel_adapter(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
    , process_count(0)
{
    this->set_name("Channel adapter");
    size_t valid = std::max(this->n_inputs, this->n_outputs) 
            % std::min(this->n_inputs, this->n_outputs);
    if(valid != 0)
        throw std::runtime_error(
                "Error in channel adapter - outputs must be even divisor of inputs. Here" 
            + std::to_string(this->n_inputs) + " " 
            + std::to_string(this->n_outputs));
    
}

template<typename Flt>
void channel_adapter<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    /*
        Find a way 
    
    */
   if(process_count == 0)
    for(size_t i = 0; i < this->n_outputs; ++i)
        std::memset(this->outputs[i], 0, sizeof(Flt) * this->bloc_size);

    if(this->n_outputs == 1)
    {
        for(size_t i = 0; i < this->bloc_size; ++i)
        {
            Flt sum = 0.0;
            for(size_t ch = 0; ch < this->n_inputs; ++ch)
            {
               sum += previous.target->outputs[ch][i]; 
            }
            this->outputs[0][i] = sum;
        }
    } else if(this->n_inputs == 1)
    {
        for(size_t ch = 0; ch < this->n_outputs; ++ch)
            std::copy(previous.target->outputs[0], 
                previous.target->outputs[0] + this->bloc_size, this->outputs[ch]);
    } else { // need to check if % is 0
        if(this->n_inputs > this->n_outputs && ((this->n_inputs % this->n_outputs) == 0 ) )  
        {
            for(size_t o = 0; o < this->n_outputs; ++o)
            {
                for(size_t s = 0; s < this->bloc_size; ++s)
                {
                    Flt sum = 0;
                    for(size_t i = o; i < this->n_inputs; i+= this->n_outputs)
                    {
                        sum += previous.target->outputs[i][s];
                    }
                    this->outputs[o][s] = sum;
                }
            }
        } else if(this->n_outputs > this->n_inputs && (this->n_outputs % this->n_inputs) == 0)
        {
            for(size_t o = 0; o < this->n_outputs; ++o)
            {
                size_t i = o % this->n_inputs;
                std::copy(previous.target->outputs[i], 
                    previous.target->outputs[i] + this->bloc_size, this->outputs[o]);
            }
        }
    }
    process_count = (process_count + 1) % this->n_nodes_in;
}

template class channel_adapter<double>;
template class channel_adapter<float>;

template<typename Flt>
upbloc<Flt>::upbloc(size_t inp , size_t outp,
        size_t blocsize, size_t samplerate)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
{
    this->set_name("Upbloc");
}

template<typename Flt>
void upbloc<Flt>::process(connection<Flt> &previous, audio_context &ctx) 
{
    size_t inp_bloc_size = previous.target->bloc_size;

    for(size_t ch = previous.output_range.first, i = previous.input_offset;
        ch <= previous.output_range.second; ++ch, ++i)
    {
        std::copy(this->outputs[i]+inp_bloc_size, 
            this->outputs[i]+this->bloc_size, this->outputs[i]);
        std::copy(previous.target->outputs[ch], 
            previous.target->outputs[ch] + inp_bloc_size, 
            this->outputs[i]+(this->bloc_size - inp_bloc_size));
    }
}

template<typename Flt>
downbloc<Flt>::downbloc(size_t inp , size_t outp,
        size_t blocsize, size_t samplerate)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
{
    this->set_name("Downbloc");
}

template<typename Flt>
void downbloc<Flt>::process(connection<Flt> &previous, audio_context &ctx) 
{
    size_t inp_bloc_size = previous.target->bloc_size;
    for(size_t ch = previous.output_range.first, i = previous.input_offset;
        ch <= previous.output_range.second; ++ch, ++i)
    {
        std::copy(previous.target->outputs[ch] + (inp_bloc_size - this->bloc_size), 
            previous.target->outputs[ch]+inp_bloc_size, this->outputs[i]);
        
    }
}

template<typename Flt>
mixer<Flt>::mixer(size_t inp, size_t outp, size_t blocsize, size_t samplerate)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
    , process_count(0)
{
    this->set_name("Mixer");
    
    for(size_t ch = 0; ch < this->n_inputs; ++ch)
        std::memset(this->outputs[ch], 0, this->bloc_size * sizeof(Flt));
}

template<typename Flt>
void mixer<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    if(this->n_nodes_in == 0) {
        throw std::runtime_error("Mixer node must have at least one input (ideally two or more, it is a mixer)");
        return;
    }

    if(this->process_count == 0)
    {
        for(size_t ch = 0; ch < this->n_outputs; ++ch)
            std::memset(this->outputs[ch], 0, this->bloc_size * sizeof(Flt));
    }

    for(size_t ch = previous.output_range.first, i = previous.input_offset; 
        ch <= previous.output_range.second; ++ch, ++i)
    {
        for(size_t n = 0; n < this->bloc_size; ++n)
            this->outputs[i][n] += 
                previous.target->outputs[ch][n];
    }

    this->process_count = (this->process_count + 1) % this->n_nodes_in;
}

template class mixer<double>;
template class mixer<float>;

template<typename Flt>
simple_upsampler<Flt>::simple_upsampler(size_t inp, size_t outp, size_t blocsize, size_t samplerate, 
        size_t order, size_t steep)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
    , f_order(order)
    , f_steep(steep)
{
    this->set_name("Simple upsampler");
    filters.resize(this->n_outputs);
    for(size_t i = 0; i < this->n_outputs; ++i)
        filters[i] = create_halfband(f_order, f_steep);
}

template<typename Flt>
simple_upsampler<Flt>::~simple_upsampler()
{
    for(size_t i = 0; i < this->n_outputs; ++i)
        destroy_halfband(filters[i]);
}

template<typename Flt>
void simple_upsampler<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    for(size_t ch = previous.output_range.first, i = previous.input_offset; 
        ch <= previous.output_range.second; ++ch, ++i)
    {
        for(size_t n = 0; n < previous.target->bloc_size; ++n)
        {
            this->outputs[i][n*2] = 
                process_halfband(filters[i], previous.target->outputs[ch][n]);
            this->outputs[i][n*2+1] = 
                process_halfband(filters[i], Flt(0.0));
        }
    }
}

template class simple_upsampler<double>;

template<typename Flt>
upsampler<Flt>::upsampler(size_t inp, size_t outp, size_t blocsize, size_t samplerate, 
        size_t num_cascade, size_t order, size_t steep)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
    , n_cascade(num_cascade)
    , f_order(order)
    , f_steep(steep)
{
    this->set_name("Upsampler");
    size_t base_samplerate = this->sample_rate / (n_cascade * 2);
    size_t base_blocsize = this->bloc_size / (n_cascade * 2);
    upsamplers.resize(num_cascade);
    
    for(size_t i = 0; i < n_cascade; ++i)
    {
        base_samplerate *= 2;
        base_blocsize *= 2;
        this->upsamplers[i] = new simple_upsampler<Flt>(inp, outp, base_blocsize, base_samplerate, 
                f_order, f_steep);
    }
}

template<typename Flt>
upsampler<Flt>::~upsampler()
{
    for(size_t i = 0; i < this->n_cascade; ++i)
        delete this->upsamplers[i];
}

template<typename Flt>
void upsampler<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    connection<Flt> p = previous;
    for(size_t i = 0; i < n_cascade; ++i) 
    {
        upsamplers[i]->process(p, ctx);
        p.target = upsamplers[i];
    }

    Flt **outputs = upsamplers.back()->outputs;
    for(size_t ch = 0; ch < this->n_outputs; ++ch)
        std::copy(outputs[ch], outputs[ch] + this->bloc_size, this->outputs[ch]);
}

template class upsampler<double>;

template<typename Flt>
downsampler<Flt>::downsampler(size_t inp, size_t outp, size_t blocsize, size_t samplerate, 
        size_t num_cascade, size_t order, size_t steep)
    : node<Flt>::node(inp, outp, blocsize, samplerate)
    , n_cascade(num_cascade)
    , f_order(order)
    , f_steep(steep)
    , n_samps_iter(n_cascade * 2)
{
    this->set_name("Downsampler");
    decimators.resize(this->n_outputs);
    for(size_t i = 0; i < this->n_outputs; ++i)
        decimators[i] = create_half_cascade(n_cascade, f_order, f_steep);
}

template<typename Flt>
downsampler<Flt>::~downsampler()
{
    for(size_t i = 0; i < this->n_outputs; ++i)
        destroy_half_cascade(decimators[i]);
}

template<typename Flt>
void downsampler<Flt>::process(connection<Flt> &previous, audio_context &ctx)
{
    for(size_t ch = previous.output_range.first, i = previous.input_offset;
        ch <= previous.output_range.second; ++ch, ++i)
    {
        for(size_t n = 0, up = 0; n < this->bloc_size; ++n, up+=n_samps_iter)
        {
            this->outputs[i][n] = process_half_cascade(this->decimators[i], 
                &(previous.target->outputs[ch][up]));
        }
    }
}

template class downsampler<double>;

/*
    Graph methods
*/

template<typename Flt>
graph<Flt>::graph(size_t inp, size_t outp, size_t blocsize, size_t samplerate) 
    : n_inputs(inp)
    , n_outputs(outp)
    , bloc_size(blocsize)
    , sample_rate(samplerate)
    , context{n_inputs, n_outputs, bloc_size, sample_rate}
{
    if(!is_power_of_two(bloc_size))
        throw std::runtime_error("Graph bloc size must be power of two");

    to_call.reserve(128);
    next_call.reserve(128);
    call_list.reserve(256);
    _mix = std::make_shared<mixer<Flt> >(outp, outp, blocsize, samplerate); 
    _input_node = std::make_shared<node<Flt>>(inp, inp, blocsize, samplerate);

    _mix->set_name("GraphOutput");
    _input_node->set_name("GraphInputs");
}

template<typename Flt>
void graph<Flt>::process_bloc()
{
    std::lock_guard<std::recursive_mutex> lock(_mtx);
    for(size_t i = 0; i < call_list.size(); ++i)
        call_list[i].callee->process(call_list[i].caller_ctx, context);
}

template<typename Flt>
void graph<Flt>::add_node(node<Flt> *n)
{
    std::lock_guard<std::recursive_mutex> lock(_mtx);
    nodes.push_back(n);
    _find_and_add_out(n);
    _generate_event_list();
}

template<typename Flt>
void graph<Flt>::add_nodes(std::vector<node<Flt>*> n)
{
    std::lock_guard<std::recursive_mutex> lock(_mtx);
    nodes.insert(nodes.end(), n.begin(), n.end());
    for(auto & it : n)
        _find_and_add_out(it);
    _generate_event_list();
}

template<typename Flt>
void graph<Flt>::remove_node(node<Flt> *n)
{
    std::lock_guard<std::recursive_mutex> lock(_mtx);
    _find_and_remove_out(n);
    _rm_node(n);
    _generate_event_list();
    return;
}

template<typename Flt>
std::string graph<Flt>::generate_patchbook_code()
{
    std::string code("");
    _generate_patchbook_code(code);
    return code;
}

template<typename Flt>
void graph<Flt>::_generate_patchbook_code(std::string &c)
{
    for(auto & it : call_list)
    {
        if(it.caller_ctx.target == nullptr)
            continue;
        for(size_t ch = it.caller_ctx.output_range.first, inch = it.caller_ctx.input_offset;
            ch <= it.caller_ctx.output_range.second; ++ch, ++inch)
        {
            std::string s = "- " + it.caller_ctx.target->get_name()
                + " (Out" + std::to_string(ch) + ") >> "
                + it.callee->get_name() 
                + "(In" + std::to_string(inch) + ")\n";
            c.append(s);
        }
    }
}

#include<map>

// Todo finish this generator for nice diagrams
template<typename Flt>
void graph<Flt>::generate_faust_diagram()
{
    std::map<std::string, std::pair<int, int> > _modules;
    for(auto & it : call_list) 
    {
        if(_modules.find(it.callee->get_name()) == _modules.end())
            _modules[it.callee->get_name()] = {it.callee->n_inputs, it.callee->n_outputs};
    }
    std::string _modules_str;
    for(auto & it : _modules)
    {
        std::string name = it.first;
        std::replace(name.begin(), name.end(), ' ', '_');
        std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char c){ return std::tolower(c); });

        std::string s = name + " = ";
        std::string sep = "";

        int rnd = rand() % 100;
        for(size_t i = 0; i < it.second.first; ++i) 
        {
            s.append(sep + " (_*" + std::to_string(rnd) + ")");
            sep = ",";
        }

        std::string op = (it.second.first > it.second.second) 
                ? ":>" : (it.second.first < it.second.second)
                ? "<:" : ":";

        if(it.second.first > 0)
            s.append(op);
        sep = "";
        rnd = rand() % 100;
        for(size_t i = 0; i < it.second.second; ++i)
        {
            s.append(sep + " (_*" + std::to_string(rnd) + ")");
            sep = ",";
        }
        s.append(";\n");
        _modules_str.append(s);

    }

    std::cout << _modules_str << std::endl;
}

template<typename Flt>
void graph<Flt>::_find_and_remove_out(node<Flt> *n)
{
    std::vector< connection<Flt> > *_connect_list = &n->connections;
    for(size_t i = 0; i < _connect_list->size(); ++i)
    {
        if(_connect_list->at(i).target == _mix.get())
        {
            _connect_list->at(i).target->disconnect(_mix.get());
        } else if(_connect_list->at(i).target->connections.size() > 0)
        {
            _find_and_remove_out(_connect_list->at(i).target);
        }
    }
}

template<typename Flt>
void graph<Flt>::_find_and_add_out(node<Flt> * n)
{
    std::vector< connection<Flt> > &_connect_list = n->connections;
    if(_connect_list.size() == 0)
    {
        if(n->n_outputs > 0) 
        {
            //std::cout << "connecting " << n->get_name() << " to mix " << std::endl;
            n->connect(_mix.get());
        }
        return;
    }
    for(size_t i = 0; i < _connect_list.size(); ++i)
    {
        /*
            The current iterated connection<Flt> has a target with 0 connections
            But the target has outputs
            Then, we connect automatically to the graph output
        */

        if(_connect_list[i].target->connections.size() == 0
            && _connect_list[i].target->n_outputs > 0)
        {
            //std::cout << "connecting " << _connect_list[i].target->get_name() << " to mix " << std::endl;
            _connect_list[i].target->connect(_mix.get()); 
            
        } else 
        /*
            The iterated target already has connections
            See if these connections correspond to the graph output (then do nothing) 
            or not (then do recursion to continue iterating the graph)
        */
        {
            bool found = false;
            for(size_t n = 0; n < _connect_list[i].target->connections.size(); ++n)
            {
                if(_connect_list[i].target->connections[n].target == _mix.get())
                {
                    found = true;
                    break;
                }
            }
            if(!found)
                _find_and_add_out(_connect_list[i].target);
        }
    }
}

template<typename Flt>
void graph<Flt>::_rm_node(node<Flt> *n)
{
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        if(n == nodes[i])
        {
            nodes.erase(nodes.begin() + i);
            return;
        }   
    }
}

template<typename Flt>
void graph<Flt>::_process_grape()
{
    std::swap(to_call_ptr, next_call_ptr);
    next_call_ptr->clear();
    if(to_call_ptr->size() == 0) 
        return;
    /*
        For each vertical (parallel) elements happening "synchronously"
    */
    for(size_t i = 0; i < to_call_ptr->size(); ++i)
    {
        connection<Flt> *caller = to_call_ptr->at(i).caller; // nullptr at first pass if no inputs
        node<Flt> *caller_ptr = (caller == nullptr) ? nullptr : caller->target;
        /*
            For each connections that will be called from the caller
        */
        if(to_call_ptr->at(i).callee == nullptr) continue;
        for(size_t j = 0; j < to_call_ptr->at(i).callee->size(); ++j)
        { 

            connection<Flt> *callee = &to_call_ptr->at(i).callee->at(j);
            call_list.push_back({callee->target, 
                { caller_ptr, callee->output_range, callee->input_offset }});
            std::vector< connection<Flt> > *nxt = &(callee->target->connections);
            next_call_ptr->push_back({callee, nxt});
        }
    }
    this->_process_grape();
}

template<typename Flt>
void graph<Flt>::_remove_duplicates()
{
    for(auto & it : call_list)
    {
        //std::cout << "full list : " << it.callee->get_name() << std::endl;
        if(it.caller_ctx.target != nullptr)
            std::cout << "\t\t called by "<<  it.caller_ctx.target->get_name() << std::endl;
    }
   for(size_t i = 0; i < call_list.size(); ++i)
   {
        for(size_t j = 0; j < call_list.size(); ++j)
        {
            if(i == j) continue;
            if(call_list[i] == call_list[j])
            {
                //std::cout << "erasing " << call_list[j].callee->get_name() << std::endl;
                if( call_list[j].caller_ctx.target != nullptr)
                    std::cout << "\t connected with " << call_list[j].caller_ctx.target->get_name() << std::endl;
                call_list.erase(call_list.begin() + j);
                --j;
            }
        }
   }
}

template<typename Flt>
void graph<Flt>::_generate_event_list()
{
    std::lock_guard<std::recursive_mutex> lock(_mtx);
    call_list.clear();
    to_call.clear();
    next_call.clear();
    /*
        Must fill the next_call (first pass) (for process_graph) with starting nodes/connections
    */
    if(_input_node->n_inputs > 0) 
    {
        _input_connection = connection<Flt>{_input_node.get()};
        next_call.push_back( call_grape{&_input_connection, &(_input_node->connections) } );
    }
    else  // Convert nodes as connections ? 
    {
        // BUG 
        _node_connections.clear();
        for(auto & it : nodes)
        {
            for(auto & cit : it->connections)
                _node_connections.push_back(
                    connection<Flt>{it, cit.output_range, cit.input_offset});
        }
        next_call.push_back( call_grape{nullptr, &_node_connections} );
    }
    to_call_ptr = &to_call;
    next_call_ptr = &next_call;
    _process_grape();
    _remove_duplicates();
}

/*
    Rtgraph engine
*/

template<typename Flt>
rtgraph<Flt>::rtgraph(size_t inp, size_t outp, size_t blocsize, size_t  samplerate)
    : graph<Flt>::graph(inp, outp, blocsize, samplerate)
{
    output_parameters.nChannels = this->n_outputs;
    output_parameters.firstChannel = 0;
    output_parameters.deviceId = dac.getDefaultOutputDevice();
    input_parameters.nChannels = this->n_inputs;
    input_parameters.firstChannel = 0;
    input_parameters.deviceId = dac.getDefaultInputDevice();
    _options = std::make_shared<RtAudio::StreamOptions>();
    _options.get()->flags = 
        RTAUDIO_NONINTERLEAVED 
        | RTAUDIO_MINIMIZE_LATENCY 
        | RTAUDIO_SCHEDULE_REALTIME;
    _options.get()->priority = 99;

    dac.showWarnings(true);
}

template<typename Flt>
void rtgraph<Flt>::list_devices()
{
    std::vector<unsigned int> ids = dac.getDeviceIds();
    RtAudio::DeviceInfo info;
    for(auto & it : ids)
    {
        info = dac.getDeviceInfo(it);
        std::cout << it << " - " << info.name << std::endl;
        std::cout << "\tID : " << info.ID << std::endl;
        std::cout << "\tinputs : " << info.inputChannels << std::endl;
        std::cout << "\toutputs : " << info.outputChannels << std::endl;
        std::cout << "\tpreferred SR : " << info.preferredSampleRate << std::endl;
    }
}

template<typename Flt>
void rtgraph<Flt>::set_devices(unsigned int input_device, unsigned int output_device)
{
    input_parameters.deviceId = input_device;
    output_parameters.deviceId = output_device;
}

template<typename Flt>
void rtgraph<Flt>::start_stream()
{
    if(dac.isStreamOpen())
    {
        std::cout << "Stream is already open" << std::endl;
        return;
    }
    RtAudio::StreamParameters *in_param_ptr = (this->n_inputs > 0) 
        ? (&input_parameters) : nullptr;
    dac.openStream(&output_parameters, in_param_ptr, RTAUDIO_FLOAT64, 
            (unsigned int)this->sample_rate, (unsigned int *)&this->bloc_size, 
            &rtgraph_callback, (void *)this, _options.get());
    dac.startStream();
}

template<typename Flt>
void rtgraph<Flt>::stop_stream()
{
    if(dac.isStreamOpen()) {
        dac.stopStream();
    }
}

int rtgraph_callback(void *out_buffer, void *in_buffer, 
        unsigned int nframes, double stream_time, RtAudioStreamStatus status, void *user_data)
{
    rtgraph<double> *_graph = (rtgraph<double> *)user_data;
    std::lock_guard<std::recursive_mutex> lock(_graph->_mtx);
    double *inputs = (double *)in_buffer;
    double *outputs = (double *) out_buffer;

    for(size_t i = 0; i < _graph->n_inputs; ++i)
    {
        // Channels are consecutive in non-interleaved RtAudio buffers
        size_t offset = _graph->bloc_size * i;
        std::copy(inputs+offset, inputs+offset + nframes, _graph->_input_node->outputs[i]);
    }
    _graph->process_bloc();
    for(size_t o = 0; o < _graph->n_outputs; ++o)
    {
        size_t offset = _graph->bloc_size * o;
        std::copy(_graph->_mix->outputs[o], 
             _graph->_mix->outputs[o] + _graph->bloc_size, outputs + offset);
    }
    return 0;
}

//#define MINIAUDIO_IMPLEMENTATION
//#include "miniaudio.h"

void  miniaudio_cbk(ma_device *device, void* outputs, 
    const void *inputs, ma_uint32 frame_count)
{
    std::cout << "MINIRT : " << frame_count << std::endl;
    mini_rtgraph<double> *_graph = (mini_rtgraph<double> *)device->pUserData;
    const float *f_inp = (const float *)inputs;
    float *f_outp = (float *)outputs;
    for(size_t i = 0; i < _graph->n_inputs; ++i)
    {
        for(size_t n = 0; n < frame_count; ++n)
            _graph->_input_node->outputs[i][n] = (double)f_inp[n*_graph->n_inputs+i];
    }
    _graph->process_bloc();
    for(size_t o = 0; o < _graph->n_outputs; ++o)
    {
        for(size_t n = 0; n < frame_count; ++n)
            f_outp[n*_graph->n_outputs+o] = (float)_graph->_mix->outputs[o][n];
    }
    std::cout << "MiniRT OK " << std::endl;
}

template<typename Flt>
mini_rtgraph<Flt>::mini_rtgraph(size_t inp, size_t outp,
        size_t blocsize, size_t samplerate)
    : graph<Flt>::graph(inp, outp, blocsize, samplerate)
{

    config = ma_device_config_init((inp>0) ? ma_device_type_duplex : ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = outp;    
    config.sampleRate = this->sample_rate;
    config.periodSizeInFrames = this->bloc_size;
    config.dataCallback = miniaudio_cbk;
    config.pUserData = this;



    if(ma_device_init(NULL, &config, &device) != MA_SUCCESS) 
        throw( std::runtime_error("Miniaudio callback could not be initialized "));
    
}

template<typename Flt>
void mini_rtgraph<Flt>::start_stream() 
{
    ma_device_start(&device);
}
template<typename Flt>
void mini_rtgraph<Flt>::stop_stream() 
{
    ma_device_uninit(&device);
}

template<typename Flt>
void mini_rtgraph<Flt>::list_devices()
{
    ma_context ctx;
    if(ma_context_init(NULL, 0, NULL, &ctx) != MA_SUCCESS)
        throw std::runtime_error("Cannot initialize Miniaudio");
    ma_device_info *playback_info;
    ma_uint32 playback_count;
    ma_device_info *capture_info;
    ma_uint32 capture_count;
    if(ma_context_get_devices(&ctx, &playback_info, &playback_count, &capture_info, &capture_count) != MA_SUCCESS)
        throw std::runtime_error("Cannot initialize Miniaudio - cannot get devices info");
    

    std::cout << "Capture : " << std::endl;
    for(size_t i = 0; i < capture_count; ++i)
    {
        std::cout  << capture_info[i].name << std::endl;
        std::cout << "\t" << capture_info[i].nativeDataFormats->channels << std::endl;
    }

    std::cout << "Playback " << std::endl;
    for(size_t i = 0; i < playback_count; ++i)
    {
        std::cout << playback_info[i].name << std::endl;
        std::cout << "\t" << playback_info[i].nativeDataFormats->channels << std::endl;
    }
}

template class mini_rtgraph<double>;