-- This is my first attempt at using Torch / Lua to build up a simple neural network
-- I will be training a simple feedforward network to be able to recognize the MNIST datastet

-- Packages
require 'torch'
require 'nn'
require 'optim'

-- Data
mnist = require 'mnist'
full_train_set = mnist.traindataset()
test_set = mnist.testdataset()
validation_set = 
{
	size = 10000,
	data = full_train_set.data[{{50001, 60000}}]:double(),
	label = full_train_set.label[{{50001, 60000}}]	
}
train_set = 
{
	size = 50000,
	data = full_train_set.data[{{1, 50000}}]:double(),
	label = full_train_set.label[{{1, 50000}}]
}

-- Model

-- The model will be a simple feedforward layer with 1 hidden unit
-- The hidden layer will have a ReLU activation function 
-- 	This was chosen because optimization is easy and quick for hidden units of the form ReLU
-- The output layer is chosen to be a softmax unit
-- 	Softmax units are useful for representing multinoulli distributions (such as the 0-9 discrete digits)
-- The loss function was chosen as cross-entropy loss function 
-- 	In general this loss function is good with nueral networks and undoes the exponential in the softmax 
-- 	activation unit

no_weights = 30
input_shape = train_set.data:size()
input_vector_size = input_shape[2] * input_shape[3] -- 28x28 for MNIST images
output_size = 10 -- Number of classes

model = nn.Sequential()
model:add(nn.Reshape(input_vector_size))
model:add(nn.Linear(input_vector_size, no_weights))
model:add(nn.ReLU())
model:add(nn.Linear(no_weights, output_size))
model:add(nn.SoftMax())

x, dL_dx = model:getParameters()

criterion = nn.CrossEntropyCriterion()

-- Optimization Algorithm

-- For the optimization algorithm, I have chosen stochastic gradient descent 
--  SGD is simple to implement and very effective for efficiently learnining the weights to a 
--  neural network
--  Possible Parameters:
-- 		learningRate: learning rate
-- 		learningRateDecay: learning rate decay
-- 		weightDecay: weigth decay
-- 		weightDecays: vector of individual weight decays
-- 		momentum: momentum
-- 		dampening: dampening for momentum
-- 		nesterov: enables Nesterov momentum

learning_rate = 0.01

sgd_params = 
{
   learningRate = learning_rate
}

-- Training

-- This function takes in the random indices for a batch and returns the loss from that batch
train_batch = function(random_indices)
	-- Create a tensors for the inputs and targets
	local inputs = torch.Tensor(random_indices:size()[1], input_shape[2], input_shape[3])
	local targets = torch.Tensor(random_indices:size())

	-- Get the random values from the training set
	for i = 1, random_indices:size()[1] do
		inputs[i] = train_set.data[random_indices[i]]
		targets[i] = train_set.label[random_indices[i]]
	end
	targets:add(1) -- No classes can == 0

	-- Create the captue to evaluate the loss function
	local feval = function(x_new)
		-- Reset data
		dL_dx:zero()
		if x ~= x_new then x:copy(x_new) end
		
		-- Perform forward step
		local forward_output = model:forward(inputs)
		local loss = criterion:forward(forward_output, targets)

		-- Perform backprop
		local gradOutput = criterion:backward(model.output, targets)
		model:backward(inputs, gradOutput)

		-- Return the values from the capture
		return loss, dL_dx
	end

	-- Run the optimization algorithm with chosen parameters
	_, fL_x = optim.sgd(feval, x, sgd_params)
	return fL_x[1]
end 

-- This function evaluates the current model on the validation set
validate = function(dataset, batch_size)
	local count = 0

	for i = 1, dataset.size, batch_size do
		local size = math.min(i + batch_size-1, dataset.size) - i
		local inputs = dataset.data[{{i, i+size-1}}]
		local targets = dataset.label[{{i, i+size-1}}]:long()
		local outputs = model:forward(inputs)
		local _, indices = torch.max(outputs, 2)
        indices:add(-1) -- Get in range of [1, 10]
		local guessed_right = indices:eq(targets):sum()
		count = count + guessed_right
	end

	return count / dataset.size 
end

-- Now we write a function to train the entire dataset
train = function(batch_size)
	batch_size = batch_size or math.floor(train_set.size / 250)
	local random_indices = torch.randperm(train_set.size)
	local batch_loss = 0
	local total_loss = 0
	local count = 0

	for i = 1, train_set.size, batch_size do
		batch_loss = train_batch(random_indices[{{i, i + batch_size - 1}}])
		total_loss = total_loss + batch_loss
		count = count + 1
	end

	return total_loss / count
end

-- Main training function
no_epochs = 300
current_epoch = 0
change_threshold = 0.001
batch_size = 200
less_accuracy_no_times = 0

while true do
	local last_accuracy = last_accuracy or 0
	current_epoch = current_epoch + 1

	local current_loss = train(batch_size)
	local accuracy = validate(validation_set, batch_size)
	print(string.format('Epoch: %d | Current Loss: %4f | Accuracy: %4f', current_epoch, current_loss, accuracy))

	if(last_accuracy > accuracy) then
		less_accuracy_no_times = less_accuracy_no_times + 1
	end

	if(math.abs(accuracy - last_accuracy) < change_threshold or current_epoch == no_epochs or less_accuracy_no_times > 1) then
		break
	end
end

print("Done training")

test_accuracy = validate(test_set, batch_size)
print(string.format('Final Test Accuracy: %4f', test_accuracy))



