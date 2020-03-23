"""failed due to insufficient resources"""

def hyper_parameters():
    lr_values = [0.001, 0.0001, 0.00005] # [0.005, 0.001, 0.0001, 0.00005, 0.00001]
    optimizers = ["RMS", "SGD", "ADAM"]
    shapes = [(504, 672)] #, (252, 336), (126, 168)]
    count = 0
    for lr in lr_values:
        for opt in optimizers:
            name = 'out' + str(count) + '_lr' + str(lr) + '_opt' + opt + '.log'
            count += 1

            sys.stdout = open('logs/' + name, "w")
            print("Running CNN with LR =", lr, ", opt = ", opt)
            for shape in shapes:
                train_gt, test_gt = gen_iterators(shape)
                model = define_cnn_model(define_optimizier(opt, lr), shape+(3,))
                print("Input shape: ", shape)
                model.fit_generator(train_gt, steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    validation_data=test_gt,
                                    validation_steps=validation_steps,
                                    workers=4)

                path_model = 'models/' + name
                save_dir = os.path.join(os.getcwd(), )
                # Save model and weights
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                model_path = os.path.join(save_dir, model_name)
                model.save(model_path)
                print('Saved trained model at %s ' % model_path)

                # Score trained model.
                scores = model.evaluate(test_gt, verbose=1)
                print('Test loss:', scores[0])
                print('Test accuracy:', scores[1])