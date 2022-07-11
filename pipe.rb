module HDLRuby::High::Std

##
# Standard HDLRuby::High library: simple pipe io.
#
########################################################################


    ## Module describing an abstract pipe input with handshake.
    #  @param typ the type of the data to input.
    system :pipe_in do |typ,clk|
        input     :in_req
        typ.input :in_val
    end


    ## Module describing an abstract pipe output with handshake.
    #  @param typ the type of the data to input.
    system :pipe_out do |typ,clk|
        output     :out_ack
        typ.output :out_val
    end


end
