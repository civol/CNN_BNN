require_relative "bneuron.rb"
require_relative "layer.rb"

## System describing a dense layer of a neural network.
#  @param  ldescr: the description of the layer.
system :dense do |ldescr|
    # Get the number of inputs.
    in_num = ldescr[:input].reduce(&:*)
    # Get the number of neurons.
    out_num = ldescr[:output].reduce(&:*)
    # Generate the input vector type.
    in_typ = bit[in_num]
    # Generate the output vector type.
    out_typ = bit[out_num]
    
    # Dense is a layer system.
    include(layer(in_typ,out_typ))

    # Generate the neurons.
    neurons = out_num.times.map do |i|
        bneuron(ldescr[:weights].flatten[i],ldescr[:biases].flatten[i],
                ldescr[:lwidth]).(:"bneuronI_#{i}").(vecX,vecY[out_num-i-1])
    end
    
end


# Unit test of the dense layer.
Unit.system :denseTest do
    # Description of a dense layer.
    descr = { type: :dense, input: [8,8], output: [4],
              weights: [
              _0000000000000000000000000000000000000000000000000000000000000000, 
              _0000000000000000000000000000000100000000000000000000000000000001,
              _0101010101010101010101010101010101010101010101010101010101010101,
              _1111111111111111111111111111111011111111111111111111111111111110
             ],
             biases: [ _1100000, _1100010, _0010100, _0001010 ],
             lwidth: 4 }

    # The test vectors.
    t_valX_expY = [
        [ _0000000000000000000000000000000000000000000000000000000000000000,[ _1, _1, _1, _0]],
        [ _0000000000000000000000000000000100000000000000000000000000000001,[ _1, _1, _1, _0]],
        [ _0101010101010101010101010101010101010101010101010101010101010101,[ _1, _1, _1, _1]],
        [ _1001110001100011100001000010000110011100011000111000010000100001,[ _1, _1, _1, _1]],
        [ _0000111100001111011100011100011100001111000011110111000111000111,[ _0, _1, _1, _1]],
        [ _1111000000111111011111000001111111110000001111110111110000011111,[ _0, _0, _1, _1]]
    ]

    # Instantiate the and connect the dense layer to test.
    inner :clk, :req, :ack
    [64].inner :vecX
    [4].inner  :vecY
    
    dense(descr).(:denseI).(clk: clk, req: req, vecX: vecX, vecY: vecY, ack: ack)

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
