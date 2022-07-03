##
#  Generic module for computing the popcount (number of bits to 1) of a
#  bitvector.
########################################################################


# Ruby function for computing the popcount value of an integer
# up to 32 bits.
def ruby_popcount(x)
    m1 = 0x55555555
    m2 = 0x33333333
    m4 = 0x0f0f0f0f
    x -= (x >> 1) & m1
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    x += x >> 8
    return (x + (x >> 16)) & 0x3f
end

# Function implementing a +n+ bit popcount of input +ivec+.
#
# NOTE: n must be no more than 32.
# $tlbn = 0
# # function :popcountLUT do |n, ivec|
# def popcountLUT(n, ivec)
#     n = n.to_i # Ensures n is an integer.
#     if n > 32 then
#         raise "Number of bits of the input #{n} is too large for a LUT popcount."
#     end
#     if n == 1 then
#       # ivec.as(bit[2])
#       ivec
#     else
#        # puts "n=#{n}, tbl will be: #{(2**n).times.map { |i| ruby_popcount(i) }}"
#        # The computation lookup table.
#        bit[n.width][-(2**n)].constant({ :"tbl#{$tlbn}" => (2**n).times.map { |i| ruby_popcount(i) } })
#        $tlbn = $tlbn + 1
# 
#        # Return the result of the computation
#        send(:"tbl#{$tlbn-1}")[ivec]
#     end
# end
# function :popcountLUT do |n, ivec|
def popcountLUT(n, ivec)
    n = n.to_i # Ensures n is an integer.
    if n > 32 then
        raise "Number of bits of the input #{n} is too large for a LUT popcount."
    end
    if n == 1 then
        ivec
    else
        # puts "n=#{n}, tbl will be: #{(2**n).times.map { |i| ruby_popcount(i) }}"
        # The computation lookup table.
        bit[n.width][-(2**n)].constant tbl? => (2**n).times.map { |i| ruby_popcount(i) }

        # Return the result of the computation
        tbl![ivec]
    end
end


# System implementing a combinatorial popcount (population count) for
# generic input bit vector +ivec+ and outputs the result to +ocnt+.
#
# Generic arguments:
# - +typ+ is the type of the input vector or the number
#   of bits of the vector.
# - +lwidth+ the input width of the LUT of the LUT layers.
system :popcount do |typ, lwidth = 4|
    # Ensure typ is a type.
    typ = bit[typ] unless typ.is_a?(Htype)
    # Compute the input width.
    iwidth = typ.width
    # Compute the output bitwidth.
    owidth = (iwidth).width
    # Compute the number of lwidth-LUT in the LUT layer.
    lnum = iwidth / lwidth
    # Comput the remaining LUT width
    lrem = iwidth % lwidth
    # puts "lwidth=#{lwidth} iwidth=#{iwidth} owidth=#{owidth} lnum=#{lnum} lrem=#{lrem}"

    # Declare the input and the output.
    typ.input       :ivec
    [owidth].output :ocnt

    # # Make the LUT layer.
    # oluts = [] # The result signals of the layer
    # # The lwidth LUTs
    # lnum.times do |i|
    #     # olut = [lwidth.width].inner :"lut#{i}"
    #     olut = [owidth].inner :"lut#{i}"
    #     # puts "range=#{((i+1)*lwidth-1)..(i*lwidth)}"
    #     # aux = [lwidth].inner :"aux#{i}"
    #     # aux <= ivec[((i+1)*lwidth-1)..(i*lwidth)]
    #     olut <= popcountLUT(lwidth,ivec[((i+1)*lwidth-1)..(i*lwidth)])
    #     # olut <= popcountLUT(lwidth,aux)
    #     oluts << olut
    # end
    # # The remaing LUT if any.
    # if lrem > 0 then
    #     # olut_last = [lrem].inner :"lut#{lnum}"
    #     olut_last = [owidth].inner :"lut#{lnum}"
    #     olut_last <= popcountLUT(lrem,ivec[(iwidth-1)..lnum*lwidth])
    #     oluts << olut_last
    # end
    # Make the LUT layer.
    oluts = [] # The result signals of the layer
    # The lwidth LUTs
    lnum.times do |i|
        olut = [owidth].inner :"lut#{i}"
        oluts << olut
    end
    # The remaing LUT if any.
    if lrem > 0 then
        olut_last = [owidth].inner :"lut#{lnum}"
        oluts << olut_last
    end
    par do
        # The lwidth LUTs
        lnum.times do |i|
            oluts[i] <= popcountLUT(lwidth,ivec[((i+1)*lwidth-1)..(i*lwidth)])
        end
        # The remaing LUT if any.
        if lrem > 0 then
            olut_last <= popcountLUT(lrem,ivec[(iwidth-1)..lnum*lwidth])
        end
    end

    # Makes the adder layer and output it.
    ocnt <= oluts.reduce(:+)
end


# System for testing the popcount.
Unit.system :popcountTest do

    # 32-bit vectors for testing and their expected results.
    # bvec = [
    #     [ _00000000000000000000000000000000, _00000 ],
    #     [ _00000000000000000000000000000001, _00001 ],
    #     [ _00000000000000000000000000000010, _00001 ],
    #     [ _10011100011000111000010000100001, _01100 ],
    #     [ _00001111000011110111000111000111, _10001 ],
    #     [ _11110000001111110111110000011111, _10100 ],
    # ]
    bvec = [
        [ _0000000000000000000000000000000000000000000000000000000000000000, _000000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001, _000010 ],
        [ _0000000000000000000000000000001000000000000000000000000000000010, _000010 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001, _011000 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111, _100010 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111, _101000 ],
    ]

    # The popcount configurations to test.
    configs = [4,6,8]

    # The popcount instances to test.
    bench = configs.map do |c|
        # ivec = [32].inner :"ivec#{c}"
        ivec = [64].inner :"ivec#{c}"
        # ovec = [6].inner :"ovec#{c}"
        ovec = [7].inner :"ovec#{c}"
        # popcount(bit[32],c).(:"popcount#{c}").(ivec,ovec)
        popcount(bit[64],c).(:"popcount#{c}").(ivec,ovec)
        [ ivec, ovec ]
    end

    # For displaying the input and expected output
    # [32].inner :ival_v
    [64].inner :ival_v
    # [6].inner :exp_v
    [7].inner :exp_v

    test do
        # Test each vector.
        bvec.each do |ival,exp|
            ival_v <= ival
            exp_v <= exp
            # On each configuration.
            bench.each do |ivec,ovec|
                ivec <= ival
            end
            !10.ns
        end
    end

end
