import os

# Set the SUMO_HOME environment variable to the path where SUMO is installed
# change it to your SUMO installation path
os.environ["SUMO_HOME"] = r"D:\Program Files (x86)\Sumo" 
# os.environ["SUMO_HOME"] = r"/usr/share/sumo"  # linux

import sumo_rl


def main():
    # load environment
    cologne3_env = sumo_rl.parallel_env(
        net_file='RESCO/cologne3/cologne3.net.xml',
        route_file='RESCO/cologne3/cologne3.rou.xml',
        use_gui=True,
        num_seconds=3600         # the number of seconds to simulate 
    )
    observations = cologne3_env.reset()
    while cologne3_env.agents:
        actions = {agent: cologne3_env.action_space(agent).sample() for agent in cologne3_env.agents}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = cologne3_env.step(actions)


if __name__ == "__main__":
    main()