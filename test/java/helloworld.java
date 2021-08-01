/* This is a simple Java program. 
FileName : "HelloWorld.java". */
/*class HelloWorld
{
	// Your program begins with a call to main().
	// Prints "Hello, World" to the terminal window.
	public static void main(String args[])
	{
		System.out.println("Hello, World");
	}
}

class HelloWorld {
	int x = 5;

	public static void main(String[] args) {
		HelloWorld myObj = new HelloWorld();
		System.out.println(myObj.x);
	}
}*/
import java.util.HashMap;

class RunnableDemo implements Runnable {
	private Thread t;
	private String threadName;

	RunnableDemo( String name) {
		threadName = name;
		System.out.println("Creating " +  threadName );
	}

	public void run() {
		System.out.println("Running " +  threadName );

		HashMap<Integer, Integer> map = new HashMap<>();

		// Adding elements to the Map
		// usiing standard add() method
		for (int i =0; i<100000; i++) {
			map.put(i, i);
		}

		// Print size and content of the Map
		System.out.println("Size of map is:- "
				+ map.size());

		/*try {
			for(int i = 4; i > 0; i--) {
				System.out.println("Thread: " + threadName + ", " + i);
				// Let the thread sleep for a while.
				Thread.sleep(50);
			}
		} catch (InterruptedException e) {
			System.out.println("Thread " +  threadName + " interrupted.");
		}*/
		System.out.println("Thread " +  threadName + " exiting.");
	}

	public void start () {
		System.out.println("Starting " +  threadName );
		if (t == null) {
			t = new Thread (this, threadName);
			t.start ();
		}
	}
}

class HelloWorld {

	public static void main(String args[]) {
		System.out.println("main ...");
		RunnableDemo R1 = new RunnableDemo( "Thread-1");
		R1.start();

		/*RunnableDemo R2 = new RunnableDemo( "Thread-2");
		R2.start();

		RunnableDemo R3 = new RunnableDemo( "Thread-1");
		R3.start();

		RunnableDemo R4 = new RunnableDemo( "Thread-2");
		R4.start();

		RunnableDemo R5= new RunnableDemo( "Thread-1");
		R5.start();

		RunnableDemo R6 = new RunnableDemo( "Thread-2");
		R6.start();

		RunnableDemo R7 = new RunnableDemo( "Thread-1");
		R7.start();

		RunnableDemo R8 = new RunnableDemo( "Thread-2");
		R8.start();

		RunnableDemo R9 = new RunnableDemo( "Thread-1");
		R9.start();

		RunnableDemo R10 = new RunnableDemo( "Thread-2");
		R10.start();*/

	}
}