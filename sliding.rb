require_relative "bneuron.rb"
require_relative "layer.rb"


## Method for computing the 1d position in a vector representing a
#  multidimentional tensor.
#  @param pos the position in the tensor
#  @param geo the geometry of the tensor
#  @return the 1d position
def pos1d(pos,geo)
    size = geo.reduce(:*)
    geo = geo[1..-1] + [1]
    pos = pos.each_with_index.reduce(0) do |sum,(x,idx)|
        (sum+x)*geo[idx]
    end
    size-pos-1
    # pos
end


## Method for gathering the connection points for a sliding block at a 
#  given position.
#  @param vecI the input vector
#  @param axis the axis to work on
#  @param in_geo the geometry of the input
#  @param in_pos the current position in the input (full position)
#  @param bk_geo the geometry of the sliding block
#  @param bk_pos the current position in the sliding block among the previous axes
def get_block_points(vecI, axis, in_geo, in_pos, bk_geo, bk_pos)
    if axis >= bk_geo.size then
        # No more axis to process, return the connection point.
        # Compute its position:
        #   current position in image + current position in the sliding block.
        pos = in_pos.zip(bk_pos).map { |x| x.reduce(&:+) }
        puts "Point pos=#{pos}"
        # Compute the position in the input vector.
        pos = pos1d(pos,in_geo)
        puts "Now point pos=#{pos}"
        # Get the connection point.
        return vecI[pos]
    else
        # Go through the axis.
        size = bk_geo[axis]
        # Recurse over the other axes.
        return size.times.map do |i|
            get_block_points(vecI,axis+1,in_geo,in_pos,bk_geo,bk_pos + [i])
        end
    end
end

## Method for generating neurons for a sliding block along a given axis.
#  @param vecI     the input vector
#  @param vecO     the output vector
#  @param k        the index of sliding block
#  @param axis     the axis to work on.
#  @param in_geo   the geometry of the input
#  @param step_geo the geometry of the slide steps
#  @param in_pos   the current position in the input among the previous axes
#  @param bk_geo   the geometry of the sliding block
#  @param out_geo  the geometry of the output
#  @param name     the base name for the HW components of the sliding block
#  @param bk_gen   the generator of the neuron:
#                  proc {|name,idx,inport,outport| ...}
def make_neurons_slide(vecI, vecO, k, axis,
                       in_geo, step_geo, in_pos, bk_geo, out_geo, 
                       name, &bk_gen)
    # puts "axis=#{axis}, in_geo=#{in_geo}, bk_geo=#{bk_geo}"
    # out_size = out_geo.reduce(&:*)
    if axis >= in_geo.size then
        # out_size = out_geo.reduce(&:*)
        out_pos =in_pos.zip(step_geo).map { |(p,s)| p/s }
        puts "Position in input: #{in_pos}."
        puts "1D position in input: #{pos1d(in_pos,in_geo)}"
        puts "Position in output: #{out_pos}."
        puts "1D position in output: #{pos1d(out_pos,out_geo)}"
        puts "output size: #{vecO.width}"
        # No more axis to process, get the connection points covered by the
        # sliding block.
        points = get_block_points(vecI, 0, in_geo, in_pos, bk_geo, [])
        # Declare the signal gathering the connection points of the sliding block
        vecF = [bk_geo.reduce(&:*)].inner :"#{name}_p"
        # Connect it.
        vecF <= points
        # Generate the neuron applying the sliding block on the current position.
        n = bk_gen.(name,k,vecF,vecO[pos1d(out_pos,out_geo)])
        # n = bk_gen.(name,k,vecF,vecO[out_size-pos1d(out_pos,out_geo)-1])
        return n
    else
        # Go through the axis.
        size = in_geo[axis]
        # Recurse over the other axes.
        return 0.step(size - bk_geo[axis],step_geo[axis]).map do |i|
            make_neurons_slide(vecI,vecO,k,axis+1,
                               in_geo,step_geo,in_pos+[i],bk_geo,out_geo,
                               name + "_#{k}_#{i}", &bk_gen)
        end
    end
end



## System describing a layer of a neural network using sliding blocks for
#  computation.
#  @param ldescr  the description of the layer.
system :sliding do |ldescr|
    # Get the number of inputs, their geometry and compute their size.
    in_geo   = ldescr[:input][1..-1]
    in_width = (in_geo.reduce(&:*))
    in_num   = ldescr[:input][0]
    in_size  = in_width*in_num
    # Get the number of outputs, their geometry and compute their size.
    out_geo  = ldescr[:output][1..-1]
    out_num  = ldescr[:output][0]
    out_width= (out_geo.reduce(&:*))
    out_size = out_width*out_num
    # Get the geometry of the sliding block.
    bk_geo   = ldescr[:blocks][1..-1]
    # Get the number of sliding blocks.
    bk_num  = ldescr[:blocks][0]
    # Get the step
    step     = ldescr[:step]
    # Get the generator of the neuron: proc {|name,idx,inport,outport| ...}
    bk_gen   = ldescr[:block_generator]
    # Get the name of the kind of sliding block.
    bk_name  = ldescr[:block_name]

    # Generate the input vector type.
    in_typ = bit[in_size]
    # Generate the output vector type.
    out_typ = bit[out_size]

    puts "Blocks with type=#{ldescr[:type]} in_width=#{in_width} and in_size=#{in_size}, out_width=#{out_width} out_size=#{out_size}"
    
    # Convolution is a layer system.
    include(layer(in_typ,out_typ))

    # Generate the neurons.
    name = bk_name.to_s
    neurons = bk_num.times do |k|
        in_num.times do |i|
            # vecO = [out_width].inner :"vecO_#{k}_#{i}"
            vecO = [out_width].inner :"vecO_#{k}_#{i}"
            vecI = [in_width].inner :"vecI_#{k}_#{i}"
            puts "Block num #{k} on input #{i}... vecX range: #{(in_num-i)*in_width-1}..#{(in_num-i-1)*in_width}, vecY range: #{((bk_num-k-1)*in_num+in_num-i)*out_width-1}..#{((bk_num-k-1)*in_num+in_num-i-1)*out_width}"
            # puts "Block num #{k} on input #{i}... vecX range: #{(in_num-i)*in_width-1}..#{(in_num-i-1)*in_width}, vecY range: #{((k)*in_num+in_num-i)*out_width-1}..#{((k)*in_num+in_num-i-1)*out_width}"
            vecI <= vecX[(in_num-i)*in_width-1..(in_num-i-1)*in_width]
            make_neurons_slide(vecI, vecO, 
                               k*in_num+i, 0, in_geo, step, [], bk_geo, out_geo,
                               name + "_#{i}", &bk_gen)
            # vecI <= vecX[(in_num-i)*in_width-1..(in_num-i-1)*in_width]
            # puts "vecO range: #{((bk_num-k-1)*in_num+in_num-i)*out_width-1}..#{((bk_num-k-1)*in_num+in_num-i-1)*out_width}"
            vecY[((bk_num-k-1)*in_num+in_num-i)*out_width-1..((bk_num-k-1)*in_num+in_num-i-1)*out_width] <= vecO
        end
    end
    
end

