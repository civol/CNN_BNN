# require "std/pipe.rb"
require_relative "pipe.rb"

## Describes a generic neural network.
system :network do |description|
    # The clock of the network.
    input :clk

    # Generate the input vector type.
    in_typ = bit[description[0][:input].reduce(&:*)]
    # Generate the output vector type.
    out_typ = bit[description[-1][:output].reduce(&:*)]

    # The input and output is inhereted from the generic parameters.
    include(pipe_in(in_typ))
    include(pipe_out(out_typ))

    # build each layer of the network.
    name = "layer_"
    layers = description.map.with_index do |ldescr,i|
        # puts "ldescr=#{ldescr}"
        send(ldescr[:type],ldescr).(name + i.to_s)
    end

    # Connect the clock the the layers.
    layers.each do |l|
        puts "layer=#{l.name}"
        l.clk <= clk
    end

    # Set up the pipeline.

    par(clk.posedge) do
        # Connect the first layer to the input.
        layers[0].vecX <= in_val
        layers[0].req  <= in_req
        # Connect the layers together
        layers.each_cons(2) do |l0,l1| 
            l1.req  <= l0.ack
            l1.vecX <= l0.vecY
        end
        # Connect the last layer to the input.
        out_val <= layers[-1].vecY
        out_ack <= layers[-1].ack
    end
end





# Unit test of the neural network.
Unit.system :networkTest do
    # Import the layers classes.
    require_relative 'convolution.rb'
    require_relative 'pooling.rb'
    require_relative 'dense.rb'

    # Description of a convolution layer.
    conv = { type: :convolution, input: [1,8,8], output: [2,6,6],
              filters: [2,3,3],
              weights: [ _000011111, _101010101 ],
              step: [1,1],
              lwidth: 4 }
    # Description of a pooling layer.
    pool = { type: :pooling, input: [2,6,6], output: [2,3,3],
              pool: [2,2],
              step: [2,2] }
    # Description of a dense layer.
    dens = { type: :dense, input: [2,3,3], output: [4],
              weights: [
              _000000000000000000, 
              _000001000000000001,
              _010101010101010101,
              _111110111111111110
             ],
             biases: [ _00000, _00010, _01010, _00101 ],
             lwidth: 4 }

    # The description of the network.
    descr = [ conv, pool, dens ]

    # The test vectors.
    t_valX_expY = [
        [ _0000000000000000000000000000000000000000000000000000000000000000,
          _0000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001,
          _0000 ],
        [ _0101010101010101010101010101010101010101010101010101010101010101,
          _0000 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001,
          _0000 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111,
          _0000 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111,
          _0000 ],
    ]

    # Instantiate the and connect the convolution layer to test.
    inner :clk, :req, :ack
    [64].inner :vecX
    [4].inner  :vecY
    
    network(descr).(:networkI).(clk: clk, in_req: req, in_val: vecX, out_val: vecY, out_ack: ack)

    # For displaying the expected value.
    [4].inner :expY

    # The test process.
    test do
        clk <= 0
        req <= 0
        !10.ns

        # Slow test.
        hprint("Slow test.\n")
        t_valX_expY.each do |(vX, eY)|
            clk <= 0
            req <= 1
            vecX <= vX
            expY <= eY
            !10.ns
            clk <= 1
            !10.ns
            clk <= 0
            req <= 0
            !10.ns
            clk <= 1
            !10.ns
        end

        # Fast test
        hprint("\nFast test.\n")
        t_valX_expY.each do |(vX, eY)|
            clk <= 0
            req <= 1
            vecX <= vX
            expY <= eY
            !10.ns
            clk <= 1
            !10.ns
        end

    end
end
