import numpy as np
import torch.nn



def createSplitLocations_equalInterval(fullTensorSize,shardSize,stride,symmetric=True):
    """
        Returns two lists of X and Y coordinates of the top-left corners of
        sub-portions (shards) of an input tensor (e.g., an image).
        Coordinates are evenly spaced according to the given stride (either
        a single number or an 1x2 list of X and Y strides, respectively).
        If "symmetric" is set to True, the coordinates will be centred around
        the input tensor (otherwise they will be aligned at the top left corner
        and hence start at zero).
    """
    if len(fullTensorSize)==1:
        fullTensorSize = [fullTensorSize,fullTensorSize]
    elif len(fullTensorSize)>2:
        fullTensorSize = fullTensorSize[len(fullTensorSize)-2:]
        
    if len(stride)==1:
        stride = [stride,stride]
        
    fullTensorSize = [float(i) for i in fullTensorSize]
    shardSize = [float(i) for i in shardSize]
    stride = [float(i) for i in stride]
    
    
    # determine min. number of shards required to cover tensor
    numShards = np.ceil(np.divide((np.subtract(fullTensorSize, shardSize)), stride)) + 1
    
    # create grid
    gridX,gridY = np.meshgrid(np.arange(numShards[0]),np.arange(numShards[1]))
    gridX *= stride[0]
    gridY *= stride[1]
    
    if symmetric:
        overhangX = np.max(gridX) + shardSize[0] - fullTensorSize[0]
        overhangY = np.max(gridY) + shardSize[1] - fullTensorSize[1]
        
        gridX -= overhangX/2
        gridY -= overhangY/2
    
    return gridX.astype(int),gridY.astype(int)
    
        
    


def createSplitLocations_auto(fullTensorSize,shardSize,stride='auto',tight=True):
    """
        Returns a matrix of X and Y locations of the top left corners
        of sub-portions (shards)of a matrix or tensor. Shards are auto-
        matically distributed evenly across the input tensor space.
        If "tight" is set to True, the split locations will be set so that
        no patch exceeds the image boundaries (but they will cover the en-
        tire image). This might result in overlapping patches, but can be
        handled by the "combineShards" function in various ways (default
        is to average the overlapping areas).
        If you wish to avoid overlapping patches, set "tight" to False.
    """
    
    if len(fullTensorSize)==1:
        fullTensorSize = [fullTensorSize,fullTensorSize]
    elif len(fullTensorSize)>2:
        fullTensorSize = fullTensorSize[len(fullTensorSize)-2:]
        
        
    if len(shardSize)==1:
        shardSize = [shardSize,shardSize]
      
    fullTensorSize = [float(i) for i in fullTensorSize]
    shardSize = [float(i) for i in shardSize]
      
        
    # create grid of shard locations
    if stride is None:
        stride = shardSize
    elif isinstance(stride, int):
        stride = (stride, stride,)
    elif isinstance(stride, float):
        if stride < 1.0:
            stride = (int(max(1, stride * shardSize[0])), int(max(1, stride * shardSize[1])))
        else:
            stride = (int(stride), int(stride))
    else:
        if isinstance(stride[0], float):
            if stride[0] < 1.0:
                stride = (int(max(1, stride[0] * shardSize[0])), int(max(1, stride[1] * shardSize[1])))
            else:
                stride = (int(stride[0]), int(stride[1]))

    if tight:
        numShards = np.ceil(np.divide(fullTensorSize, shardSize)).clip(1)
        totalSize = numShards * shardSize
        overhang = totalSize - fullTensorSize
        startLoc = [0,0]
        maxStride = np.ceil(shardSize - (overhang/(numShards-1).clip(1))).clip(1, shardSize)
        stride = (min(stride[0], maxStride[0]), min(stride[1], maxStride[1]))
    else:
        numShards = np.ceil(np.divide(fullTensorSize, stride)).clip(1)
        totalSize = numShards * stride
        overhang = totalSize - fullTensorSize
        startLoc = np.ceil(overhang/2)
        
    numShards = np.ceil(np.divide(fullTensorSize, stride)).clip(1)

    coordsX = torch.arange(numShards[0]) * stride[0] - startLoc[0]
    coordsY = torch.arange(numShards[1]) * stride[1] - startLoc[1]
    
    if tight:
        # manually shift last coordinates to make sure they don't cross the image width and height
        coordsX[-1] = fullTensorSize[0] - shardSize[0]
        coordsY[-1] = fullTensorSize[1] - shardSize[1]
    
    gridX,gridY = torch.meshgrid(coordsX,coordsY)

    return gridX.long(),gridY.long()
    
    
    
def splitTensor(inputTensor,shardSize,locX,locY):
    """
        Divides an input tensor into sub-tensors at given locations.
        The locations determine the top left corners of the sub-tensors.
        If the locations exceed the input tensor's boundaries, it is
        padded with zeros.
    """
    if len(inputTensor.size())>3:
        inputTensor = torch.squeeze(inputTensor)
        
    sz = inputTensor.size()
    
    locX = locX.long()
    locY = locY.long()
    
    startLocX = torch.min(locX)
    startLocY = torch.min(locY)
    endLocX = torch.max(locX) + shardSize[0]
    endLocY = torch.max(locY) + shardSize[1]
    
    
    # pad tensor with zeros
    if startLocX<0 or startLocY<0 or endLocX>sz[1] or endLocY>sz[2]:
        padL = int(torch.abs(startLocX))
        padT = int(torch.abs(startLocY))
        padR = int(endLocX - sz[1])
        padB = int(endLocY - sz[2])
        tensor = torch.nn.ZeroPad2d((padT,padB,padL,padR))(inputTensor.unsqueeze(0))
        tensor = tensor.data.squeeze()
        
        # shift locations accordingly
        locX = locX + padL
        locY = locY + padT
    else:
        tensor = inputTensor
    
    
    # crop tensor
    if locX.dim()==2:
        numPatches = locX.size(0)*locX.size(1)
        locX = locX.view(-1)
        locY = locY.view(-1)
    else:
        numPatches = locX.size(0)

    result = torch.Tensor(numPatches,sz[0],shardSize[0],shardSize[1]).type(tensor.dtype).to(tensor.device)
    for x in range(0,numPatches):
        result[x,:,:,:] = tensor[:,locX[x]:locX[x]+shardSize[0],locY[x]:locY[x]+shardSize[1]]
    
    return result



def combineShards(shards,locX,locY,outSize,overlapRule):
    """
        Combines a series of shards (sub-images) composed in a tensor (NxCxWxH),
        with N = #shards, C = #bands, W and H = width and height of the shards,
        respectively.
        locX and locY denote the top left X and Y coordinates of each shard.
        The number of X and Y locations must match the number of shards.
        Shards are restored to a full tensor, which is then centre-cropped to
        the specified "outSize" (optional; this accounts for shards exceeding the original
        image boundaries).
        Elements in zones of overlapping shards are treated according to the spe-
        cified "overlapRule":
        - "average": patch contents are equally averaged
        - "max": the maximum value is retained
        - "min": the minimum value is chosen
        - "sum": the sum is calculated along all overlapping patches
    """
    
    locX = locX.astype(int)
    locY = locY.astype(int)
    
    sz = shards.size()
    
    startLocX = np.min(locX)
    startLocY = np.min(locY)
    
    endLocX = np.max(locX) + sz[2] + np.abs(startLocX)
    endLocY = np.max(locY) + sz[3] + np.abs(startLocY)
    
    if startLocX<0:
        locX += np.abs(startLocX)
    if startLocY<0:
        locY += np.abs(startLocY)
        
    
    # prepare output tensor as well as counting grid
    out = torch.Tensor(sz[1],int(endLocX),int(endLocY))
    out[:] = 0
    count = torch.Tensor(1,int(endLocX),int(endLocY))
    count[:] = 0
    
    # iterate over shards and restore
    for i in range(0,sz[0]):
        posX = int(i / locX.shape[1])
        posY = int(i % locX.shape[1])
        
        coordsX = locX[posX,posY]
        coordsY = locY[posX,posY]
        
        shard = shards[i,:,:,:]
        
        if overlapRule=='sum':
            out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]] += shard
        elif overlapRule=='max':
            out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]] = torch.max(out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]],shard)
        elif overlapRule=='min':
            out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]] = torch.min(out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]],shard)
        else:
            out[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]] += shard
            count[:,coordsX:coordsX+sz[2],coordsY:coordsY+sz[3]] += 1

    
    
    # normalise according to specified flag
    if overlapRule=='average' or overlapRule=='avg':
        out /= count.expand_as(out)
        
        
    # crop if necessary
    if outSize is not None:
        sz_out = out.size()
        if sz_out[1]!=outSize[0] or sz_out[2]!=outSize[1]:
            overhangX = (sz_out[1] - outSize[0])/2
            overhangY = (sz_out[2] - outSize[1])/2
            
            out = out[:,overhangX:-overhangX,overhangY:-overhangY]
        
    return out