require_relative "bneuron.rb"
require_relative "sliding.rb"



## System describing a pooling layer of a neural network.
#  @param ldescr: the description of the layer.
system :pooling do |ldescr|
    # A pooling layer is a kind of sliding block layer.
    # Update the description accordingly and make it a subclass module
    # of module sliding.
    # ldescr[:blocks] = [ldescr[:input][0], *ldescr[:pool]]
    ldescr[:blocks] = [1, *ldescr[:pool]]
    ldescr[:block_name] = "pool"
    # Set the step as the pooling size, and the pad to 0.
    ldescr[:step] = ldescr[:pool].clone
    ldescr[:pad]  = ldescr[:pool].map { 0 }
    # Create the generator.
    ldescr[:block_generator] = proc do |name,k,inport,outport|
        outport <= inport.reduce(&:|)
    end

    # Make the module a specific subclass of sliding
    include(sliding(ldescr))
end



# Unit test of the pooling layer.
Unit.system :poolingTest do
    # Description of a dense layer.
    descr = { type: :pooling, input: [1,4,4,4], output: [1,2,2,2],
              pool: [2,2,2],
              step: [2,2,2] }

    # The test vectors.
    t_valX_expY = [
        [ _0000000000000000000000000000000000000000000000000000000000000000,
          _00000000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001,
          _00000000 ],
        [ _0101010101010101010101010101010101010101010101010101010101010101,
          _00000000 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001,
          _00000000 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111,
          _00000000 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111,
          _00000000 ],
    ]

    # Instantiate the and connect the convolution layer to test.
    inner :clk, :req, :ack
    [64].inner :vecX
    [24].inner  :vecY
    
    pooling(descr).(:poolingI).(clk: clk, req: req, vecX: vecX, vecY: vecY, ack: ack)

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
