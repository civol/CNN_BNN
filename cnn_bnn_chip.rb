require 'json.rb'

require 'std/hruby_unit.rb'

require_relative 'cnn_bnn.rb'



# System describing a chip containing a CNN with BNN.
system :cnn_bnn_chip do

    # The number of samples (assumed to be a power of 2).
    nsamples = 2**3


    # The test vectors.
    t_valX_expY = samples_from_json("samples.json").sample(nsamples)
    # Compute the input and output width from the samples.
    iwidth = t_valX_expY[0][0].width
    owidth = t_valX_expY[0][1].width

    # The input and output of the circuit.
    input :clk, :rst
    input :req
    output :ack
    [owidth].output :vecY, :expY

    # Instantiate the and connect the convolution layer to test.
    [iwidth].inner :vecX
    # [owidth].inner  :vecY
    
    network_from_json("cnn_bnn","cnn_state.json").(clk: clk, in_req: req, in_val: vecX, out_val: vecY, out_ack: ack)

    # The memory containing the input vectors.
    bit[iwidth][-nsamples].constant vecXs: t_valX_expY.map {|s| s[0] }
    # The memory containing the expected ouputs
    bit[owidth][-nsamples].constant expYs: t_valX_expY.map {|s| s[1] }

    # For displaying the expected value.
    # [owidth].inner :expY

    # The sample index.
    [nsamples.width].inner :idx

    # The execution process.
    par(clk.posedge) do
        hif(rst) { idx <= 0 }
        helse do
            vecX <= vecXs[idx]
            expY <= expYs[idx]
            idx <= idx + 1
        end
    end
end
