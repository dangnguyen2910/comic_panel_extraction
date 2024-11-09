import torch 

def get_unet(in_channels, out_channels) -> 'model': 
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                           in_channels=in_channels, out_channels=out_channels, 
                           init_features=32, pretrained=False)
    
    return model


def train_one_epoch(model, train_dataloader, loss_fn, optimizer, device='cpu'):
    running_loss = 0
    last_loss = 0

    model.train()
    for i, data in enumerate(train_dataloader):
        imgs, masks = data
        imgs = imgs.to(device).float()
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    


def validate(model, validation_dataloader, loss_fn, device='cpu'):
    running_vloss = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            imgs, masks = data
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()

            voutputs = model(imgs)
            vloss = loss_fn(voutputs, masks)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'Validation loss: {avg_vloss}')



def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, epochs, device='cpu'):
    for epoch in range(epochs): 
        print(f'Epoch {epoch + 1}')
        print('-' * 20)

        train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        validate(model, validation_dataloader, loss_fn, device)

    return model

        



def predict(model, img, device) -> torch.tensor: 
    img = img.to(device).float()
    model.eval()
    pred = model(img)
    return pred

    