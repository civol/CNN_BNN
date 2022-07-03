##
#  Generic module for computing the activation function of a
#  binarized neural network combining optimized bnorm and sign computation.
#
#  See: "On-Chip Memory Based Binarized Convolutional Deep Neural Network 
#  Applying Batch Normalization Free Technique on an FPGA" by
#  Haruyoshi Yonekawa and Hiroki Nakahara from Tokyo Institute of Technology,
#  Japan for details.
###########################################################################



# System implementing a combinatorial simplified bnorm and sign computation
# with input integer +x+ and bias +b+ and bnorm weight and ouputs a binarized
# 0 for -1 and 1 for +1 values.
#
# Generic arguments:
# - +typ+ is the type of the input integer.
# # - +threshold+ is the threshold changing result from -1 to +1
system :bact do |typ|
    # Ensure typ is a type.
    typ = bit[typ] unless typ.is_a?(Htype)
    # Create the computation type.
    styp = signed[typ.width+1]

    # The input and outputs.
    typ.input :x, :b
    output :z

    # Compute the y value.
    styp.inner :y
    y <= x.as([typ.width+1]) + [b[typ.width-1],b].to_expr

    # Compute and output the sign.
    z <= ~y[typ.width]
end


# System for testing the bnorm and sign computation.
Unit.system :bactTest do

    # 8-bit integers for testing and their expected results.
    bench = [
        [ _s00000000, 
          _s00000000, _0 ],
        [ _s00000001,
          _s00000000, _0 ],
        [ _s00000001,
          _s00000001, _0 ],
        [ _s00010000,
          _s10000000, _1 ],
        [ _s00010000,
          _s00000001, _0 ],
        [ _s11110000,
          _s11111111, _1 ],
    ]

    # The input and output signals.
    [8].inner :x, :w
    inner :z

    # The bact instance to test.
    bact(bit[8]).(:bact).(x,w,z)

    # For displaying the input and expected output
    signed[8].inner :x_v, :w_v
    inner :z_v

    test do
        # Test each vector.
        bench.each do |bx, bw, bz|
            x_v <= bx
            w_v <= bw
            z_v <= bz
            x <= bx
            w <= bw
            !10.ns
        end
    end

end
