import torch

def training(model, data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.NLLLoss()
    losses = []

    for epoch in range(20):
        total_loss = 0
        for batch_idx, (context, target) in enumerate(data_loader):

            model.zero_grad()
            preds = model.forward(context)
            loss = loss_function(preds, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss)

        print('total_loss:', total_loss)