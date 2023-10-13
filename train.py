import torch
import time

# ------------------------------------- #
# Define training parameters
# ------------------------------------- #
LR = 0.01  # learning rate
optimizer = None
loss_function = None

def train(epoch, data_loader, model):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (label, text) in enumerate(data_loader):
        optimizer.zero_grad(True)
        predicted_label = model(text)
        loss = loss_function(predicted_label, label.squeeze_())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        # total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_acc += (predicted_label == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed time {:2.2f}s".format(epoch, idx, len(data_loader), 
                                                                                                        total_acc / total_count, 
                                                                                                        elapsed))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(data_loader, model):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad(): # very important, if not we update the gradient
        for idx, (label, text) in enumerate(data_loader):
            predicted_label = model(text)
            loss = loss_function(predicted_label, label.squeeze_())
            # total_acc += (predicted_label.argmax(1) == label).sum().item() # If we have one output possible class
            total_acc += (predicted_label == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def train_model(epochs, train_dataloader, validation_data_loader, model, total_accu=None):
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch, train_dataloader, model)
        accu_val = evaluate(validation_data_loader, model)
        print("-" * 59)
        print("| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val))
        print("-" * 59)

def predict(model, data_loader):
    predicted_label = []
    with torch.no_grad(): # very important, if not we update the gradient
        for idx, (label, text) in enumerate(data_loader):
            predicted_label.append(model(text).argmax(1))
    return torch.cat(predicted_label)
