require 'json.rb'

require_relative 'network.rb'

# Import the layers classes.
require_relative 'convolution.rb'
require_relative 'pooling.rb'
require_relative 'dense.rb'

# The possible types of layers.
LAYER_TYPES = [ "dense", "convolution", "pooling" ]

## Binarize recursively the content of an array.
#  @param arr the array whose content is to binarize.
#  @return arr
def binarize_vectorize!(arr)
    arr.map! do |sub|
        if sub[0].is_a?(Array) then
            binarize_vectorize!(sub)
        else
            sub.map {|e| e >= 0 ? 1 : 0 }.join.to_value
        end
    end
end


## sample builder from a json description obtained from FastNeurons.
#  @param fname the name of the json file to load.
#  @return an instance of the resulting network.
def samples_from_json(fname)
    # Load the description from the json file.
    samples = []
    File.open(fname,"r") { |f| samples = JSON.load(f) }
    # Process it to match the HW CNN library.
    samples.each { |sample| binarize_vectorize!(sample) }
    return samples
end



## BNN CNN builder from a json description obtained from FastNeurons.
#  @param name  the name of the instance of the network.
#  @param fname the name of the json file to load.
#  @return an instance of the resulting network.
def network_from_json(name,fname)
    descr = []
    # Load the description from the json file.
    File.open(fname,"r") { |f| descr = JSON.load(f) }
    # Ensure the keys are symbols.
    descr.map! { |layer| layer.transform_keys! {|k| k.to_sym } }
    # Process it to match the HW CNN library.
    descr = descr.select do |layer|
        # Process the type.
        layer[:type] = layer[:type].downcase
        # Process the weights if any.
        weights = layer[:weights]
        puts "First weights=#{weights}"
        # One set of weights.
        width = 1
        if weights then
            binarize_vectorize!(weights)
            width = weights.flatten[0].width
        end
        puts "weights=#{weights.inspect}"
        # Process the biases if any.
        biases = layer[:biases]
        puts "first biases=#{biases.inspect}"
        # biases.map! {|b| b.round - width / 2 } if biases
        biases.map! {|b| b.round - ((width+0.1) / 2).round } if biases
        puts "biases=#{biases.inspect}"
        # Add the LUT width if not.
        layer[:lwidth] = 1 unless layer[:lwidth]
        # Keep the layer if supported by the HW.
        LAYER_TYPES.include?(layer[:type].to_s)
    end

    # puts "descr=#{descr}"

    # Generate the network instance.
    network(descr).(name.to_s)
end



# Unit test of the neural network.
Unit.system :network_from_jsonTest do


    # The test vectors.
    t_valX_expY = samples_from_json("samples.json")
    # Compute the input and output width from the samples.
    iwidth = t_valX_expY[0][0].width
    owidth = t_valX_expY[0][1].width

    # Instantiate the and connect the convolution layer to test.
    inner :clk, :req, :ack
    [iwidth].inner :vecX
    [owidth].inner  :vecY
    
    network_from_json("cnn_bnn","cnn_state.json").(clk: clk, in_req: req, in_val: vecX, out_val: vecY, out_ack: ack)

    # For displaying the expected value.
    [owidth].inner :expY

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
