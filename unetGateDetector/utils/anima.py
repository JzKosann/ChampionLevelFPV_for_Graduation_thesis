import time
import os

def anima():
    frames = [
        # 平稳燃烧的阶段
        r"""
             ~~~~~~
           //      \\
          ||        ||
          ||        ||
          ||  ~~~   ||    
          ||  ~~~   ||
          ||        ||
         /  \______/  \\
        /______________\\
         |            |
         |    炼丹炉    |
         |____________|
        """,
        r"""
             ~~~~~~
           //      \\
          ||        ||
          ||  ~~~   ||
          ||  ~~~~~~||    
          ||  ~~~~~~||
          ||        ||
         /  \______/  \\
        /______________\\
         |            |
         |    炼丹炉    |
         |____________|
        """,
        r"""
             ~~~~~~
           //      \\
          ||        ||
          || ~~~~~~ ||
          || ~~~~~~~||    
          || ~~~~~~~||
          ||        ||
         /  \______/  \\
        /______________\\
         |            |
         |   炼丹炉     |
         |____________|
        """,
        # 炉子开始震动的阶段
        r"""
             ~~~~~~
           //      \\
          ||        ||
          ||~~~~~~~ ||
          ||~~~~~~~ ||    
          || ~~~~~~ ||
          ||        ||
         /  \______/  \\
        /______________\\
         |    炉子震动  |
         |   炼丹炉     |
         |____________|
        """,
        r"""
             ~~~~~~
           //      \\
          ||        ||
          || ~~~~~~~||
          ||~~~~~~~~||    
          ||~~~~~~~~||
          ||        ||
         /  \______/  \\
        /______________\\
         |   炉子裂开  |
         |   炼丹炉     |
         |____________|
        """,
        # 火焰强烈且炉子裂开的阶段
        r"""
            ~~~~~~~~
           //       \\
          ||         ||
          || ~~~~~~~~||
          ||~~~~~~~~~||    
          ||~~~~~~~~~||
          ||   炉子裂开  ||
         /  \______/  \\
        /______________\\
         |    火焰四射  |
         |   炼丹炉     |
         |____________|
        """,
        # 炸开的帧
        r"""
           炸开了!!!!
           *  *  *  *
        *     *  *    *
      *  *  * *  * *  *  *
        *   *   *   * *
        """,
        r"""
           BOOOOM!!!!
           *  *  *  *
        *     *  *    *
      *  *  * *  * *  *  *
        *   *   *   * *
        """,
        r"""
           BOOOOM!!!!
           *  *  *  *
        *     *  *    *
      *  *  * *  * *  *  *
        *   *   *   * *
        """
    ]

    # 火焰逐渐增强，炉子震动，裂开
    for i in range(3):  # 火焰增强部分
        for frame in frames[:3]:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(frame)
            time.sleep(0.4)

    for i in range(3):  # 炉子震动和裂开部分
        for frame in frames[3:6]:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(frame)
            time.sleep(0.3)

    # 最后炸开
    for _ in range(5):
        for frame in frames[6:]:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(frame)
            time.sleep(0.2)

def train_complete_printf():
    print("********************************************")
    time.sleep(0.5)
    print("*                                          *")
    time.sleep(0.5)
    print("*  🎉🎉🎉 Attention! The impossible has been done! 🎉🎉🎉  *")
    time.sleep(0.5)
    print("*                                          *")
    print("********************************************")
    time.sleep(0.5)
    print("")
    print("🔥 The training is finally over! 🔥")
    print("")
    time.sleep(0.5)
    print("💾💾💾 Your precious model has been carefully sealed 💾💾💾")
    time.sleep(0.5)
    print("✨ In a magical place known as: 'best_model.pth' ✨")
    time.sleep(0.5)
    print("")
    print("🚀🚀 Now go forth, brave coder, and unleash its power upon the world! 🚀🚀")
    time.sleep(0.5)
    print("")
    print("But remember...")
    time.sleep(0.8)
    print("🔒 Keep it safe. Keep it secret. Or face the bugs of doom. 💀")
    print("")
    time.sleep(2)

