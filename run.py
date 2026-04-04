import subprocess



import time


import sys


import os


from dotenv import load_dotenv





def main():


    load_dotenv()  # Load environment variables from .env file


    print("==================================================")


    print(" Starting MetroMind WhatsApp Agent Stack")


    print("==================================================\n")





    # Define paths based on the current working directory


    base_dir = os.path.dirname(os.path.abspath(__file__))


    bridge_dir = os.path.join(base_dir, "whatsapp_bridge")


    bridge_script = os.path.join(bridge_dir, "src", "server.js")


    agent_script = os.path.join(base_dir, "agent", "agent.py")





    bridge_process = None


    agent_process = None





    try:


        # 1. Start the Node.js Bridge in the background


        print("[System] Booting Node.js WhatsApp Bridge...")


        bridge_process = subprocess.Popen(


            ["node", bridge_script],


            cwd=bridge_dir,


            stdout=sys.stdout,  # Pipe bridge logs directly to the main terminal


            stderr=sys.stderr


        )





        # Give the Express server a moment to bind to the port


        time.sleep(3)


        print("\n[System] Booting Python Agno Agent...\n")





        # 2. Start the Python Agent in the foreground


        agent_process = subprocess.Popen(


            [sys.executable, agent_script],


            cwd=os.path.join(base_dir, "agent")


        )





        # Keep the main script alive while the agent runs


        agent_process.wait()





    except KeyboardInterrupt:


        print("\n\n[System] Shutdown signal received. Terminating processes...")


    


    finally:


        # Clean up processes on exit to prevent port hoarding/zombie processes


        if agent_process and agent_process.poll() is None:


            agent_process.terminate()


            print("[System] Terminated Python Agent.")


            


        if bridge_process and bridge_process.poll() is None:


            bridge_process.terminate()


            print("[System] Terminated Node.js Bridge.")


            


        print("[System] Stack fully shut down.")





if __name__ == "__main__":


    main()