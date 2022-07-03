require_relative "bmac.rb"
require_relative "bact.rb"

##
#  Generic module for computing a binarized neuron.
########################################################################

$count = 0


# System implementing a combinatorial binarized neuron for
# generic input bit vector +vecX+ and outputs the result to binarized +a+.
#
# NOTE: binarized values are 0 for -1 and 1 for +1
#
# Generic arguments:
# - +weights+ the weights value
# - +bias+ the bias value
# - +lwidth+ the input width of the LUT of the LUT layers of the popcount.
system :bneuron do |weights, bias, lwidth = 4|
    # Compute the input type of the neuron.
    typ = bit[weights.width]
    # Compute the input width.
    iwidth = typ.width
    # Compute the linear result width
    zwidth = iwidth.width

    # The input and output.
    typ.input :vecX
    output :a

    puts "lwidth=#{lwidth} weights.width=#{weights.width} count=#{$count}"
    $count += 1

    typ.constant vecW: weights
    [zwidth].constant b: bias

    # The result of the mac computation.
    [zwidth].inner :z

    # The binarized mac.
    bmac(typ,lwidth).(:bmacI).(vecX,vecW,z)

    # The activation function.
    bact(bit[zwidth]).(:actI).(z,b,a)
end


# System for testing the binarized mac.
Unit.system :bneuronTest do

    # The weights and biases
    t_w_b = [
        [ _00000000000000000000000000000000, _000000 + _110000 ],
        [ _00000000000000000000000000000001, _000001 + _110000 ],
        [ _01010101010101010101010101010101, _001010 + _110000 ],
        [ _11111111111111111111111111111110, _000101 + _110000 ]
    ]
    # t_w_b = [
    #     [ _0000000000000000000000000000000000000000000000000000000000000000, _0000000 ],
    #     [ _0000000000000000000000000000000100000000000000000000000000000001, _0000010 ],
    #     [ _0101010101010101010101010101010101010101010101010101010101010101, _0010100 ],
    #     [ _1111111111111111111111111111111011111111111111111111111111111110, _0001010 ]
    # ]

    # 32-bit vectors for testing and their expected results.
    t_valX_expA = [
        [ _00000000000000000000000000000000,[ _1, _1, _1, _0]],
        [ _00000000000000000000000000000001,[ _1, _1, _1, _0]],
        [ _01010101010101010101010101010101,[ _1, _1, _1, _1]],
        [ _10011100011000111000010000100001,[ _1, _1, _1, _1]],
        [ _00001111000011110111000111000111,[ _0, _1, _1, _1]],
        [ _11110000001111110111110000011111,[ _0, _0, _1, _1]]
    ]
    # # 64-bit vectors for testing and their expected results.
    # t_valX_expA = [
    #     [ _0000000000000000000000000000000000000000000000000000000000000000,[ _1, _1, _1, _0]],
    #     [ _0000000000000000000000000000000100000000000000000000000000000001,[ _1, _1, _1, _0]],
    #     [ _0101010101010101010101010101010101010101010101010101010101010101,[ _1, _1, _1, _1]],
    #     [ _1001110001100011100001000010000110011100011000111000010000100001,[ _1, _1, _1, _1]],
    #     [ _0000111100001111011100011100011100001111000011110111000111000111,[ _0, _1, _1, _1]],
    #     [ _1111000000111111011111000001111111110000001111110111110000011111,[ _0, _0, _1, _1]]
    # ]


    # The bneurons instances to test.
    bench = t_w_b.map.with_index do |(w,b),i|
        vecX = [32].inner :"vecX#{i}"
        # vecX = [64].inner :"vecX#{i}"
        a = inner :"a#{i}"
        bneuron(bit[32],w,b).(:"bneuron#{i}").(vecX,a)
        # bneuron(bit[64],w,b).(:"bneuron#{i}").(vecX,a)
        [ vecX, a ]
    end

    # For displaying the input and expected output
    [32].inner :vecX_v
    # [64].inner :vecX_v
    [4].inner :expAs_v

    test do
        # Test each vector.
        t_valX_expA.each do |valX, expAs|
            vecX_v <= valX
            expAs_v <= expAs
            # On each configuration.
            bench.each do |vecX, a|
                vecX <= valX
            end
            !10.ns
        end
    end

end
