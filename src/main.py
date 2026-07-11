import time

from eyes import Eyes
from ears import Ear

ear = Ear(train_epochs=50)

# ear.run()
# time.sleep(60 * 10)
# ear.stop()
ear.train()
