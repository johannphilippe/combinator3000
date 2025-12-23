# Combinator 3000

This project development moved to Xrune : github.com/johannphilippe/xrune

Audio graph generator for multi-environment combination.
Experimental / in development.

## Mechanism 

Main component is `node`. A Node is an audio processor. 
It has :

    - A number of inputs and outputs
    - An output buffer for samples processing (input buffer is always owned by previous node)
    - A `process` method (compute)
    - A vector of `connection` (represent output connections
    - A `connect` method to connect a node to another

Nodes can be put in a `graph` object or a `rtgraph`. 
A `graph` handles a list of `node` (starting of chain) and generates an internal list of events to trigger, 
automatically finding which node is connected to which. 
Nodes must be connected together BEFORE the "starting node" of the chain is put in the graph.

## Dependencies 

Some nodes won't compile unless you have installed : 
- Faust (with libfaust and LLVM support)
- Csound 
- libsndfile

