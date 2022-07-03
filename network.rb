require "std/pipe.rb"

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
        send(ldescr[:type],ldescr).(name + i.to_s)
    end

    # Connect the clock the the layers.
    layers.each { |l| l.clk <= clk }

    # Connect the first layer to the input.
    layers[0].vecX <= in_val
    layers[0].req  <= in_req
    # Connect the layers together.
    layers.each_cons(2) do |l0,l1| 
        l1.req  <= l0.ack
        l1.vecX <= l0.vecY
    end
    # Connect the last layer to the input.
    out_val <= layers[-1].vecY
    out_ack <= layers[-1].ack
end
