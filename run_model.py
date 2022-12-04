from tqdm import tqdm
import torch

def run_model(model, optimizer, train_loader, test_loader, args, epoch_num = 10, run_eval = True):
    """
    usage:
    just run training for particles:
        l, pp = run_model(model, optimizer, train_loader, test_loader, args, 1, False)
    run training for losses:
        l, el, pp, epp = run_model(model, optimizer, train_loader, test_loader, args, 30, True)
    """
    losses = []
    eval_losses = []

    for epoch in tqdm(range(epoch_num)):
        per_e_loss = []
        # print("going to train")
        model.train() #just a toggle switch

        # print("starting iterations")
        for iteration, data in enumerate(train_loader):

            output, window = data
            model.zero_grad()
            loss, log_loss, particle_pred = model.step(
                window, output, args)
            loss.backward()

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            loss_last = log_loss.to('cpu').detach().numpy()
            loss_all = loss.to('cpu').detach().numpy()
            per_e_loss.append([loss_last, loss_all])
        losses.append(per_e_loss)
        
        if run_eval:
            model.eval()
            
            eval_loss_last = []
            with torch.no_grad():
                for iteration, data in enumerate(test_loader):
                    output, window = data

                    model.zero_grad()
                    loss, log_loss, eval_pred = model.step(
                        window, output, args)

                    eval_loss_last.append([loss.to('cpu').detach().numpy(), log_loss.to('cpu').detach().numpy()])
            
            eval_losses.append(eval_loss_last)

    if run_eval:
        return losses, eval_losses, particle_pred, eval_pred
    return losses, particle_pred