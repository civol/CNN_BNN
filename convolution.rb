require_relative "bneuron.rb"
require_relative "layer.rb"

# idee:
#     recurse_filter(axis,in_geo,in_pos,fil_geo,fil_pos)
#        if axis >= fil_pos.size then
#            return fil_pos
#        else
#            size = in_fil[axis]
#            return size.times.map do |i|
#                recurse_filter(axis+1,in_geo,in_pos,fil_geo,fil_pos + [i])
#            end
#        end
# 
#     recurse_slide(axis,in_geo,in_pos,fil_geo,weights,lwidth,name)
#        if axis >= in_geo.size then
#            points = recurse_filter(0,in_geo,in_pos,fil_geo,fil_pos,weights,lwidth,name)
#            vecF = [fil_geo.reduce(&:*)].inner :"#{name}_p"
#            n = bneuron(vecF.typ,weights).(:"#{name}_f").(vecF,vecY[pos1d(in_pos,in_geo)])
#            vecF <= points
#            return n
# 
#        else
#            size = in_geo[axis]
#            return (size - fil_geo[axis]).times.map do |i|
#                recurse_slide(axis+1,in_geo,in_pos+[i],fil_geo,weights,lwidth,name + "_#{i}")
#            end
#        end

## Method for computing the 1d position in a vector representing a
#  multidimentional tensor.
#  @param pos the position in the tensor
#  @param geo the geometry of the tensor
#  @return the 1d position
def pos1d(pos,geo)
    geo = geo[1..-1] + [1]
    pos = pos.each_with_index.reduce(0) do |sum,(x,idx)|
        (sum+x)*geo[idx]
    end
end


## Method for gathering the connection points for a filter at a given position.
#  @param vecI the input vector
#  @param axis the axis to work on
#  @param in_geo the geometry of the input
#  @param in_pos the current position in the input (full position)
#  @param fil_geo the geometry of the filter
#  @param fil_pos the current position in the filter among the previous axes
def get_filter_points(vecI, axis, in_geo, in_pos, fil_geo, fil_pos)
    if axis >= fil_geo.size then
        # No more axis to process, return the connection point.
        # Compute its position:
        #   current position in image + current position in the filter.
        pos = in_pos.zip(fil_pos).map { |x| x.reduce(&:+) }
        # puts "pos=#{pos}"
        # Compute the position in the input vector.
        pos = pos1d(pos,in_geo)
        # puts "Now pos=#{pos}"
        # Get the connection point.
        return vecI[pos]
    else
        # Go through the axis.
        size = fil_geo[axis]
        # Recurse over the other axes.
        return size.times.map do |i|
            get_filter_points(vecI,axis+1,in_geo,in_pos,fil_geo,fil_pos + [i])
        end
    end
end

## Method for generating neurons for a sliding filter along a given axis.
#  @param vecI the input vector
#  @param vecO the output vector
#  @param axis the axis to work on.
#  @param in_geo the geometry of the input
#  @param in_pos the current position in the input among the previous axes
#  @param fil_geo the geometry of the filter
#  @param out_geo the geometry of the output
#  @param weigths the weights of the filter
#  @param lwidth the width of the LUTs used in the popcount circuit
def make_neurons_slide(vecI,vecO,axis,in_geo,in_pos,fil_geo,out_geo,
                       weights,lwidth,name)
    # puts "axis=#{axis}, in_geo=#{in_geo}, fil_geo=#{fil_geo}"
    if axis >= in_geo.size then
        puts "Position in input: #{in_pos}."
        puts "1D position in input: #{pos1d(in_pos,in_geo)}"
        puts "1D position in output: #{pos1d(in_pos,out_geo)}"
        # No more axis to process, get the connection points covered by the
        # filter.
        points = get_filter_points(vecI, 0, in_geo, in_pos, fil_geo, [])
        # Declare the signal gathering the connection points of the filter.
        vecF = [fil_geo.reduce(&:*)].inner :"#{name}_p"
        # Connect it.
        vecF <= points
        # Declare the neuron applying the filter on the current position.
        n = bneuron(weights,-vecF.width/2,lwidth).(:"#{name}_f").
        # n = bneuron(weights,0,lwidth).(:"#{name}_f").
            (vecF,vecO[pos1d(in_pos,out_geo)])
        return n
    else
        # Go through the axis.
        size = in_geo[axis]
        # Recurse over the other axes.
        return (size - fil_geo[axis] + 1).times.map do |i|
            make_neurons_slide(vecI,vecO,axis+1,
                               in_geo,in_pos+[i],fil_geo,out_geo,
                               weights,lwidth, name+"_#{i}")
        end
    end
end



## System describing a convolution layer of a neural network.
#  @param ldescr: the description of the layer.
system :convolution do |ldescr|
    # Get the number of inputs, their geometry and compute their size.
    in_geo   = ldescr[:input][1..-1]
    in_num   = ldescr[:input][0]
    in_size  = (in_geo.reduce(&:*))*in_num
    # Get the number of outputs, their geometry and compute their size.
    out_geo  = ldescr[:output][1..-1]
    out_num  = ldescr[:output][0]
    out_width= (out_geo.reduce(&:*))
    out_size = out_width*out_num
    # Get the weights.
    weights  = ldescr[:weights]
    # Get the geometry of the filters.
    fil_geo  = ldescr[:filters][1..-1]
    # Get the number of filters.
    fil_num  = ldescr[:filters][0]
    # Get the width of the luts.
    lwidth   = ldescr[:lwidth]

    # Generate the input vector type.
    in_typ = bit[in_size]
    # Generate the output vector type.
    out_typ = bit[out_size]
    
    # Convolution is a layer system.
    include(layer(in_typ,out_typ))

    # Generate the neurons.
    name = "filter_"
    neurons = fil_num.times do |k|
        vecO = [out_width].inner :"vecO_#{k}"
        puts "Filter num #{k}..."
        make_neurons_slide(vecX, vecO, 
                           0, in_geo, [], fil_geo, out_geo,
                           weights[k],
                           lwidth, name + "_#{k}")
        vecY[k*out_width..(k+1)*out_width-1] <= vecO
    end
    
end



# Unit test of the convolution layer.
Unit.system :convolutionTest do
    # Description of a dense layer.
    descr = { type: :convolution, input: [1,4,4,4], output: [2,2,2,3],
              filters: [2,3,3,2],
              weights: [ _000000000111111111, _010101010101010101 ],
              lwidth: 4 }

    # The test vectors.
    t_valX_expY = [
        [ _0000000000000000000000000000000000000000000000000000000000000000,
          _000000000000000000000000 ],
        [ _0000000000000000000000000000000100000000000000000000000000000001,
          _000000000000000000000000 ],
        [ _0101010101010101010101010101010101010101010101010101010101010101,
          _000000000000000000000000 ],
        [ _1001110001100011100001000010000110011100011000111000010000100001,
          _000000000000000000000000 ],
        [ _0000111100001111011100011100011100001111000011110111000111000111,
          _000000000000000000000000 ],
        [ _1111000000111111011111000001111111110000001111110111110000011111,
          _000000000000000000000000 ],
    ]

    # Instantiate the and connect the convolution layer to test.
    inner :clk, :req, :ack
    [64].inner :vecX
    [24].inner  :vecY
    
    convolution(descr).(:convolutionI).(clk: clk, req: req, vecX: vecX, vecY: vecY, ack: ack)

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
