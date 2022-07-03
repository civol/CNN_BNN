require_relative "popcount.rb"
# require_relative "popcount_flat.rb"

##
#  Generic module for computing a binarized mac (number of bits to 1) of a
#  bitvector.
########################################################################



# System implementing a combinatorial binarized mac for
# generic input bit vector +vecA+ and +vecB+ and outputs the result to +vecP+.
#
# Generic arguments:
# - +typ+ is the type of the input vectors or the number
#   of bits of the vectors.
# - +lwidth+ the input width of the LUT of the LUT layers of the popcount.
system :bmac do |typ, lwidth = 4|
    # Ensure typ is a type.
    typ = bit[typ] unless typ.is_a?(Htype)
    # Compute the input width.
    iwidth = typ.width
    # Compute the output bitwidth.
    owidth = (iwidth).width

    # The input and outputs.
    typ.input :vecA, :vecB
    [owidth].output :vecP

    # The XNOR layer.
    [iwidth].inner :xnors
    xnors <= ~(vecA ^ vecB)

    # The popcount layer.
    popcount(typ,lwidth).(:acc).(xnors,vecP)
end


# System for testing the binarized mac.
Unit.system :bmacTest do

    # 32-bit vectors for testing and their expected results.
    # bvec = [
    #     [ _00000000000000000000000000000000, 
    #       _00000000000000000000000000000000, _100000 ],
    #     [ _00000000000000000000000000000001,
    #       _00000000000000000000000000000001, _100000 ],
    #     [ _01010101010101010101010101010101,
    #       _10101010101010101010101010101010, _000000 ],
    #     [ _10011100011000111000010000100001,
    #       _00001111000011110111000111000111, _001101 ],
    #     [ _00001111000011110111000111000111,
    #       _11110000001111110111110000011111, _001111 ],
    #     [ _11110000001111110111110000011111,
    #       _11111111110000000000001111111111, _001010 ],
    # ]
    bvec = [
        [ _0000000000000000000000000000000000000000000000000000000000000000, 
          _0000000000000000000000000000000000000000000000000000000000000000, _1000000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001,
          _0000000000000000000000000000000100000000000000000000000000000001, _1000000 ],
        [ _0101010101010101010101010101010101010101010101010101010101010101,
          _1010101010101010101010101010101010101010101010101010101010101010, _0000000 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001,
          _0000111100001111011100011100011100001111000011110111000111000111, _0011010 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111,
          _1111000000111111011111000001111111110000001111110111110000011111, _0011110 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111,
          _1111111111000000000000111111111111111111110000000000001111111111, _0010100 ],
    ]

    # The bmac configurations to test.
    configs = [4,6,8]

    # The bmac instances to test.
    bench = configs.map do |c|
        # ivecA = [32].inner :"ivecA#{c}"
        ivecA = [64].inner :"ivecA#{c}"
        # ivecB = [32].inner :"ivecB#{c}"
        ivecB = [64].inner :"ivecB#{c}"
        # ovec = [6].inner :"ovec#{c}"
        ovec = [7].inner :"ovec#{c}"
        # bmac(bit[32],c).(:"bmac#{c}").(ivecA,ivecB,ovec)
        bmac(bit[64],c).(:"bmac#{c}").(ivecA,ivecB,ovec)
        [ ivecA, ivecB, ovec ]
    end

    # For displaying the input and expected output
    # [32].inner :ivalA_v, :ivalB_v
    [64].inner :ivalA_v, :ivalB_v
    # [6].inner :exp_v
    [7].inner :exp_v

    test do
        # Test each vector.
        bvec.each do |ivalA, ivalB, exp|
            ivalA_v <= ivalA
            ivalB_v <= ivalB
            exp_v <= exp
            # On each configuration.
            bench.each do |ivecA, ivecB, ovec|
                ivecA <= ivalA
                ivecB <= ivalB
            end
            !10.ns
        end
    end

end
