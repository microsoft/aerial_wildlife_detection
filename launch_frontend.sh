# Convenience function to run just the frontend.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger

python runserver.py --settings_filepath=settings_windowCropping.ini --instance=UserHandler,FileServer,LabelUI