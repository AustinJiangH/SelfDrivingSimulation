import time
import torch

# from helper_evaluation import compute_accuracy


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='minibatch_loss', resume=False, resume_epoch=0):

    if resume:
        print("Resuming status...")
        checkpoint = torch.load(f"/model-{str(resume_epoch)}.h5",
                                map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    start_time = time.time()
    train_loss_list, valid_loss_list = [], []
    
    model.to(device)
    for epoch in range(num_epochs):
        

        model.train()
        for batch_index, (centers, lefts, rights) in enumerate(train_loader):

            minibatch_loss = 0
            # transfer data to device 
            centers[0] = centers[0].to(device)
            centers[1] = centers[1].to(device)
            lefts[0] = lefts[0].to(device)
            lefts[1] = lefts[1].to(device)
            rights[0] = rights[0].to(device)
            rights[1] = rights[1].to(device)

            # forward and backward propagation
            optimizer.zero_grad()
            datas = [centers, lefts, rights]
            for data in datas:
                imgs, angles = data
                outputs = model(imgs)
                loss = torch.nn.MSELoss()(outputs, angles.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                minibatch_loss += loss.item()
                
            train_loss_list.append(minibatch_loss)
            scheduler.step()


            if not batch_index % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_index:04d}/{len(train_loader):04d} '
                      f'| Loss: {minibatch_loss:.4f}')

        # validation 
        model.eval()
        with torch.no_grad():  # save memory during inference
            for batch_index, (centers, lefts, rights) in enumerate(valid_loader):

                minibatch_loss = 0
                # transfer data to device 
                centers[0] = centers[0].to(device)
                centers[1] = centers[1].to(device)
                lefts[0] = lefts[0].to(device)
                lefts[1] = lefts[1].to(device)
                rights[0] = rights[0].to(device)
                rights[1] = rights[1].to(device)

                ## FORWARD AND BACK PROP
                optimizer.zero_grad()
                datas = [centers, lefts, rights]
                for data in datas:
                    imgs, angles = data
                    outputs = model(imgs)
                    loss = torch.nn.MSELoss()(outputs, angles.unsqueeze(1).float())
                    minibatch_loss += loss.item()


                valid_loss_list.append(minibatch_loss)
            

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train Loss: {train_loss_list[-1] :.5f} '
                  f'| Validation Loss: {valid_loss_list[-1] :.5f}')


        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(train_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

        # Save model
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        torch.save(state, f'model-{state["epoch"]}.h5')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')


    return train_loss_list, valid_loss_list
