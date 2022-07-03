## Abstract system describing a layer of a neural network.
#  @param  typX: the data type of the input vector.
#  @param  typY: the data type of the output vector.
#  @input  clk the input clock.
#  @input  req the input request signal.
#  @input  vecX the input vector.
#  @output ack the output acknowledge (computation completes).
#  @output vecY the output vector.
system :layer do |typX, typY|
    input :clk, :req
    typX.input :vecX
    output :ack
    typY.output :vecY

    # The synchronization part.
    # By default, layer is assumed to require only one cycle to complete.
    sub(:sync) do
        par(clk.posedge) do
            ack <= req
        end
    end
end
