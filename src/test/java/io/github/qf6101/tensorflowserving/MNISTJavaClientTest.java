package io.github.qf6101.tensorflowserving;


/**
 * Created by qfeng on 17-4-10.
 */
public class MNISTJavaClientTest {
    public static void main(String[] args) throws Exception {
        // Initialize the server's host and port
        String host = "127.0.0.1";
        int port = 9000;
        // Create MNIST client
        MNISTJavaClient client = new MNISTJavaClient(host, port);
        // Predict a batch of MNIST test images
        // Download the mnist test image and label files in advance
        client.predict("mnist_test_data/t10k-images-idx3-ubyte",
                "mnist_test_data/t10k-labels-idx1-ubyte");
        // Shut down the connection to server
        client.shutdown();
    }
}
