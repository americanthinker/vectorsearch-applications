#Disable memory paging and swapping performance on the host to improve performance. 
sudo sudo swapoff -a

#Increase the number of memory maps available to OpenSearch. 
sudo echo 'vm.max_map_count=262144' >> /etc/sysctl.conf

# Reload the kernel parameters using sysctl
sudo sysctl -p

#check the new value of vm.max_map_count
cat /proc/sys/vm/max_map_count