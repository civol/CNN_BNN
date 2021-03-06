require_relative "bneuron.rb"
require_relative "sliding.rb"



## System describing a convolution layer of a neural network.
#  @param ldescr: the description of the layer.
system :convolution do |ldescr|
    # A convolution layer is a kind of sliding block layer.
    # Update the description accordingly and make it a subclass module
    # of module sliding.
    ldescr[:blocks] = ldescr[:filters].clone
    ldescr[:block_name] = "filter"
    # Get the weights of the filter and the width of the LUT in the popcount.
    weights  = ldescr[:weights]
    # puts "weights=#{weights}"
    lwidth   = ldescr[:lwidth]
    # Use them to create the generator.
    ldescr[:block_generator] = proc do |name,k,inport,outport|
        # puts "weights[#{k}]=#{weights[k].inspect}"
        bneuron(weights[k],-(((inport.width+0.1)/2).round),lwidth).(:"#{name}_f").
        # bneuron(weights[k],-((inport.width/2).round),lwidth).(:"#{name}_f").
            (inport,outport)
    end

    # Make the module a specific subclass of sliding
    include(sliding(ldescr))
end



# Unit test of the convolution layer.
Unit.system :convolutionTest do
    # Description of a convolution layer.
    descr = { type: :convolution, input: [1,4,4,4], output: [2,2,2,3],
              filters: [2,3,3,2],
              weights: [ _000000000111111111, _010101010101010101 ],
              step: [1,1,1],
              lwidth: 4 }

    # The test vectors.
    t_valX_expY = [
        [ _0000000000000000000000000000000000000000000000000000000000000000,
          _000000000000000000000000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001,
          _000000000000000000000000 ],
        [ _0101010101010101010101010101010101010101010101010101010101010101,
          _000000000000000000000000 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001,
          _000000000000000000000000 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111,
          _000000000000000000000000 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111,
          _000000000000000000000000 ],
    ]

    # Instantiate the and connect the convolution layer to test.
    inner :clk, :req, :ack
    [64].inner :vecX
    [24].inner  :vecY
    
    convolution(descr).(:convolutionI).(clk: clk, req: req, vecX: vecX, vecY: vecY, ack: ack)

    # For displaying the expected value.
    [24].inner :expY

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
